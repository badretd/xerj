//! A small backtracking regex engine with the semantics of Python `re` for the patterns of the frozen decline policy.
//!
//! Why not the `regex` crate: the frozen patterns use lookaround, and a leftmost-first DFA does not give `re`'s match on lazy
//! patterns with lookaround without another engine beside it; `\w`, `\s` and `re.I` differ from Python's as well. Here the pattern
//! text is copied from the Python file unchanged and interpreted with the same rules: ordered alternatives, greedy and lazy
//! repeats with backtracking, lookahead, one-character lookbehind, CPython's character tables. No dependency is added.
//!
//! Every scan has a step budget: an adversarial question ends with [`OverBudget`] instead of running for minutes. The caller fails
//! closed (a declined answer, code D2X_RULE_ERROR). The frozen Python has no budget; no bench question comes near it.

use super::pystr::{ignorecase_canon, is_digit, is_space, is_word};
use std::cell::Cell;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverBudget;

/// steps one question may use over all its scans
pub const DEFAULT_BUDGET: u64 = 200_000_000;

pub struct Budget(Cell<u64>);

impl Budget {
    pub fn new(steps: u64) -> Budget {
        Budget(Cell::new(steps))
    }
    #[inline]
    fn spend(&self, n: u64) -> Result<(), OverBudget> {
        let v = self.0.get();
        if v < n {
            return Err(OverBudget);
        }
        self.0.set(v - n);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Cat {
    Word,
    NotWord,
    Digit,
    NotDigit,
    Space,
    NotSpace,
}

impl Cat {
    #[inline]
    fn has(self, c: char) -> bool {
        match self {
            Cat::Word => is_word(c),
            Cat::NotWord => !is_word(c),
            Cat::Digit => is_digit(c),
            Cat::NotDigit => !is_digit(c),
            Cat::Space => is_space(c),
            Cat::NotSpace => !is_space(c),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
struct Class {
    neg: bool,
    ranges: Vec<(char, char)>,
    cats: Vec<Cat>,
}

impl Class {
    #[inline]
    fn has(&self, c: char) -> bool {
        let m = self.ranges.iter().any(|&(a, b)| a <= c && c <= b)
            || self.cats.iter().any(|k| k.has(c));
        m != self.neg
    }
}

#[derive(Clone, Debug)]
enum Node {
    Empty,
    Char(char),
    Class(Class),
    Any,
    Cat(Vec<Node>),
    Alt(Vec<Node>),
    Repeat {
        node: Box<Node>,
        min: u32,
        max: Option<u32>,
        greedy: bool,
    },
    Group(usize, Box<Node>),
    Look {
        ahead: bool,
        neg: bool,
        node: Box<Node>,
    },
    WordB,
    Start,
    End,
    StrictEnd,
}

#[derive(Clone, Debug)]
enum Inst {
    Char(char),
    CharI(u32),
    Class(u32),
    Any,
    Split(usize, usize),
    Jmp(usize),
    Save(usize),
    WordB,
    Start,
    End,
    StrictEnd,
    Look { ahead: bool, neg: bool, body: usize },
    Match,
}

struct Parser<'a> {
    p: &'a [char],
    i: usize,
    groups: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.p.get(self.i).copied()
    }
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn starts(&mut self, s: &str) -> bool {
        let cs: Vec<char> = s.chars().collect();
        if self.p[self.i..].starts_with(&cs) {
            self.i += cs.len();
            true
        } else {
            false
        }
    }
    fn alt(&mut self) -> Node {
        let mut alts = vec![self.cat()];
        while self.eat('|') {
            alts.push(self.cat());
        }
        if alts.len() == 1 {
            alts.pop().unwrap()
        } else {
            Node::Alt(alts)
        }
    }
    fn cat(&mut self) -> Node {
        let mut v = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let a = self.atom();
            v.push(self.quant(a));
        }
        match v.len() {
            0 => Node::Empty,
            1 => v.pop().unwrap(),
            _ => Node::Cat(v),
        }
    }
    fn number(&mut self) -> Option<u32> {
        let s = self.i;
        let mut n = 0u32;
        while let Some(d) = self.peek().and_then(|c| c.to_digit(10)) {
            n = n * 10 + d;
            self.i += 1;
        }
        (self.i > s).then_some(n)
    }
    fn quant(&mut self, a: Node) -> Node {
        let (min, max) = match self.peek() {
            Some('*') => {
                self.i += 1;
                (0, None)
            }
            Some('+') => {
                self.i += 1;
                (1, None)
            }
            Some('?') => {
                self.i += 1;
                (0, Some(1))
            }
            Some('{') => {
                self.i += 1;
                let lo = self.number().unwrap_or(0);
                let hi = if self.eat(',') {
                    self.number()
                } else {
                    Some(lo)
                };
                assert!(self.eat('}'), "pyre: bad repeat");
                (lo, hi)
            }
            _ => return a,
        };
        let greedy = !self.eat('?');
        assert!(
            max.is_some() || !nullable(&a),
            "pyre: an unbounded repeat of a nullable body"
        );
        Node::Repeat {
            node: Box::new(a),
            min,
            max,
            greedy,
        }
    }
    fn esc_cat(c: char) -> Option<Cat> {
        Some(match c {
            'w' => Cat::Word,
            'W' => Cat::NotWord,
            'd' => Cat::Digit,
            'D' => Cat::NotDigit,
            's' => Cat::Space,
            'S' => Cat::NotSpace,
            _ => return None,
        })
    }
    fn atom(&mut self) -> Node {
        let c = self.peek().expect("pyre: pattern end");
        self.i += 1;
        match c {
            '(' => {
                let n = if self.starts("?:") {
                    self.alt()
                } else if self.starts("?=") {
                    self.look(true, false)
                } else if self.starts("?!") {
                    self.look(true, true)
                } else if self.starts("?<=") {
                    self.look(false, false)
                } else if self.starts("?<!") {
                    self.look(false, true)
                } else {
                    assert!(self.peek() != Some('?'), "pyre: unsupported group");
                    self.groups += 1;
                    let g = self.groups;
                    Node::Group(g, Box::new(self.alt()))
                };
                assert!(self.eat(')'), "pyre: unbalanced group");
                n
            }
            '[' => self.class(),
            '.' => Node::Any,
            '^' => Node::Start,
            '$' => Node::End,
            '\\' => {
                let e = self.peek().expect("pyre: pattern end");
                self.i += 1;
                if e == 'b' {
                    Node::WordB
                } else if let Some(k) = Self::esc_cat(e) {
                    Node::Class(Class {
                        neg: false,
                        ranges: Vec::new(),
                        cats: vec![k],
                    })
                } else {
                    assert!(!e.is_ascii_alphanumeric(), "pyre: unsupported escape");
                    Node::Char(e)
                }
            }
            _ => Node::Char(c),
        }
    }
    fn look(&mut self, ahead: bool, neg: bool) -> Node {
        let n = self.alt();
        assert!(ahead || width_one(&n), "pyre: lookbehind of width 1 only");
        Node::Look {
            ahead,
            neg,
            node: Box::new(n),
        }
    }
    fn class(&mut self) -> Node {
        let mut k = Class {
            neg: self.eat('^'),
            ..Class::default()
        };
        let mut first = true;
        loop {
            let c = self.peek().expect("pyre: unterminated class");
            self.i += 1;
            if c == ']' && !first {
                break;
            }
            first = false;
            let lo = if c == '\\' {
                let e = self.peek().expect("pyre: pattern end");
                self.i += 1;
                if let Some(cat) = Self::esc_cat(e) {
                    k.cats.push(cat);
                    continue;
                }
                assert!(!e.is_ascii_alphanumeric(), "pyre: unsupported escape");
                e
            } else {
                c
            };
            if self.peek() == Some('-') && self.p.get(self.i + 1).is_some_and(|&x| x != ']') {
                self.i += 1;
                let mut hi = self.peek().unwrap();
                self.i += 1;
                if hi == '\\' {
                    hi = self.peek().unwrap();
                    self.i += 1;
                }
                k.ranges.push((lo, hi));
            } else {
                k.ranges.push((lo, lo));
            }
        }
        Node::Class(k)
    }
}

fn nullable(n: &Node) -> bool {
    match n {
        Node::Char(_) | Node::Class(_) | Node::Any => false,
        Node::Empty
        | Node::Look { .. }
        | Node::WordB
        | Node::Start
        | Node::End
        | Node::StrictEnd => true,
        Node::Cat(v) => v.iter().all(nullable),
        Node::Alt(v) => v.iter().any(nullable),
        Node::Repeat { node, min, .. } => *min == 0 || nullable(node),
        Node::Group(_, x) => nullable(x),
    }
}

fn width_one(n: &Node) -> bool {
    match n {
        Node::Char(_) | Node::Class(_) | Node::Any => true,
        Node::Group(_, x) => width_one(x),
        Node::Alt(v) => v.iter().all(width_one),
        _ => false,
    }
}

pub struct Regex {
    prog: Vec<Inst>,
    classes: Vec<Class>,
    slots: usize,
}

struct Compiler {
    prog: Vec<Inst>,
    classes: Vec<Class>,
    icase: bool,
    looks: Vec<(usize, Node)>,
}

impl Compiler {
    fn emit(&mut self, i: Inst) -> usize {
        self.prog.push(i);
        self.prog.len() - 1
    }
    fn node(&mut self, n: &Node) {
        match n {
            Node::Empty => {}
            Node::Char(c) => {
                if self.icase {
                    self.emit(Inst::CharI(ignorecase_canon(*c)));
                } else {
                    self.emit(Inst::Char(*c));
                }
            }
            Node::Class(k) => {
                // a class under re.I would need case closure: none of the frozen patterns has a letter in one
                assert!(
                    !self.icase || k.ranges.is_empty(),
                    "pyre: a character class under IGNORECASE"
                );
                let id = self.classes.len() as u32;
                self.classes.push(k.clone());
                self.emit(Inst::Class(id));
            }
            Node::Any => {
                self.emit(Inst::Any);
            }
            Node::Cat(v) => v.iter().for_each(|x| self.node(x)),
            Node::Alt(v) => {
                let mut jumps = Vec::new();
                for (i, x) in v.iter().enumerate() {
                    if i + 1 < v.len() {
                        let s = self.emit(Inst::Split(0, 0));
                        self.node(x);
                        jumps.push(self.emit(Inst::Jmp(0)));
                        let next = self.prog.len();
                        self.prog[s] = Inst::Split(s + 1, next);
                    } else {
                        self.node(x);
                    }
                }
                let end = self.prog.len();
                for j in jumps {
                    self.prog[j] = Inst::Jmp(end);
                }
            }
            Node::Repeat {
                node,
                min,
                max,
                greedy,
            } => {
                for _ in 0..*min {
                    self.node(node);
                }
                match max {
                    None => {
                        let s = self.emit(Inst::Split(0, 0));
                        self.node(node);
                        self.emit(Inst::Jmp(s));
                        let end = self.prog.len();
                        self.prog[s] = if *greedy {
                            Inst::Split(s + 1, end)
                        } else {
                            Inst::Split(end, s + 1)
                        };
                    }
                    Some(m) => {
                        let mut splits = Vec::new();
                        for _ in *min..*m {
                            splits.push(self.emit(Inst::Split(0, 0)));
                            self.node(node);
                        }
                        let end = self.prog.len();
                        for s in splits {
                            self.prog[s] = if *greedy {
                                Inst::Split(s + 1, end)
                            } else {
                                Inst::Split(end, s + 1)
                            };
                        }
                    }
                }
            }
            Node::Group(g, x) => {
                self.emit(Inst::Save(2 * g));
                self.node(x);
                self.emit(Inst::Save(2 * g + 1));
            }
            Node::Look { ahead, neg, node } => {
                let at = self.emit(Inst::Look {
                    ahead: *ahead,
                    neg: *neg,
                    body: 0,
                });
                self.looks.push((at, (**node).clone()));
            }
            Node::WordB => {
                self.emit(Inst::WordB);
            }
            Node::Start => {
                self.emit(Inst::Start);
            }
            Node::End => {
                self.emit(Inst::End);
            }
            Node::StrictEnd => {
                self.emit(Inst::StrictEnd);
            }
        }
    }
}

/// one match: `span` in code points, groups as `(start, end)`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Match {
    pub start: usize,
    pub end: usize,
    pub groups: Vec<Option<(usize, usize)>>,
}

enum Bt {
    Branch(usize, usize),
    Restore(usize, usize),
}

const NONE: usize = usize::MAX;

impl Regex {
    pub fn new(pattern: &str, icase: bool) -> Regex {
        let cs: Vec<char> = pattern.chars().collect();
        let mut p = Parser {
            p: &cs,
            i: 0,
            groups: 0,
        };
        let ast = p.alt();
        assert!(p.i == cs.len(), "pyre: trailing pattern text");
        Self::from_ast(&ast, p.groups, icase)
    }

    /// the pattern for `fullmatch`: use it with [`Regex::match_at_start`]
    pub fn full(pattern: &str) -> Regex {
        let cs: Vec<char> = pattern.chars().collect();
        let mut p = Parser {
            p: &cs,
            i: 0,
            groups: 0,
        };
        let ast = p.alt();
        assert!(p.i == cs.len(), "pyre: trailing pattern text");
        Self::from_ast(&Node::Cat(vec![ast, Node::StrictEnd]), p.groups, false)
    }

    /// `(?<![\w]) + re.escape(literal) + (?![\w])` (roles.mention_spans)
    pub fn word_bounded_literal(literal: &str, icase: bool) -> Regex {
        let word = Class {
            neg: false,
            ranges: Vec::new(),
            cats: vec![Cat::Word],
        };
        let mut v = vec![Node::Look {
            ahead: false,
            neg: true,
            node: Box::new(Node::Class(word.clone())),
        }];
        v.extend(literal.chars().map(Node::Char));
        v.push(Node::Look {
            ahead: true,
            neg: true,
            node: Box::new(Node::Class(word)),
        });
        Self::from_ast(&Node::Cat(v), 0, icase)
    }

    fn from_ast(ast: &Node, groups: usize, icase: bool) -> Regex {
        let mut c = Compiler {
            prog: Vec::new(),
            classes: Vec::new(),
            icase,
            looks: Vec::new(),
        };
        c.node(ast);
        c.emit(Inst::Match);
        while let Some((at, n)) = c.looks.pop() {
            let body = c.prog.len();
            c.node(&n);
            c.emit(Inst::Match);
            if let Inst::Look { body: b, .. } = &mut c.prog[at] {
                *b = body;
            }
        }
        Regex {
            prog: c.prog,
            classes: c.classes,
            slots: 2 * groups + 2,
        }
    }

    /// run from `pc` at `at`; the end position on success
    fn run(
        &self,
        s: &[char],
        pc0: usize,
        at: usize,
        caps: &mut [usize],
        budget: &Budget,
    ) -> Result<Option<usize>, OverBudget> {
        let mut stack: Vec<Bt> = Vec::new();
        let (mut pc, mut pos) = (pc0, at);
        loop {
            budget.spend(1)?;
            let ok = match &self.prog[pc] {
                Inst::Char(c) => {
                    if s.get(pos) == Some(c) {
                        pos += 1;
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                Inst::CharI(k) => {
                    if s.get(pos).is_some_and(|&x| ignorecase_canon(x) == *k) {
                        pos += 1;
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                Inst::Class(k) => {
                    if s.get(pos)
                        .is_some_and(|&x| self.classes[*k as usize].has(x))
                    {
                        pos += 1;
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                Inst::Any => {
                    if s.get(pos).is_some_and(|&x| x != '\n') {
                        pos += 1;
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                Inst::Split(a, b) => {
                    stack.push(Bt::Branch(*b, pos));
                    pc = *a;
                    true
                }
                Inst::Jmp(a) => {
                    pc = *a;
                    true
                }
                Inst::Save(k) => {
                    stack.push(Bt::Restore(*k, caps[*k]));
                    caps[*k] = pos;
                    pc += 1;
                    true
                }
                Inst::WordB => {
                    let a = pos > 0 && is_word(s[pos - 1]);
                    let b = pos < s.len() && is_word(s[pos]);
                    if a != b {
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                Inst::Start => {
                    if pos == 0 {
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                Inst::End => {
                    if pos == s.len() || (pos + 1 == s.len() && s[pos] == '\n') {
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                Inst::StrictEnd => {
                    if pos == s.len() {
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                Inst::Look { ahead, neg, body } => {
                    let m = if *ahead {
                        self.run(s, *body, pos, caps, budget)?.is_some()
                    } else if pos == 0 {
                        false
                    } else {
                        self.run(s, *body, pos - 1, caps, budget)? == Some(pos)
                    };
                    if m != *neg {
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                Inst::Match => return Ok(Some(pos)),
            };
            if ok {
                continue;
            }
            loop {
                match stack.pop() {
                    None => return Ok(None),
                    Some(Bt::Restore(k, v)) => caps[k] = v,
                    Some(Bt::Branch(a, p)) => {
                        pc = a;
                        pos = p;
                        break;
                    }
                }
            }
        }
    }

    fn at(&self, s: &[char], start: usize, b: &Budget) -> Result<Option<Match>, OverBudget> {
        let mut caps = vec![NONE; self.slots];
        Ok(self.run(s, 0, start, &mut caps, b)?.map(|end| Match {
            start,
            end,
            groups: (1..self.slots / 2)
                .map(|g| {
                    let (a, z) = (caps[2 * g], caps[2 * g + 1]);
                    (a != NONE && z != NONE).then_some((a, z))
                })
                .collect(),
        }))
    }

    /// `pattern.search(s, from)`
    pub fn search_from(
        &self,
        s: &[char],
        from: usize,
        b: &Budget,
    ) -> Result<Option<Match>, OverBudget> {
        for start in from..=s.len() {
            if let Some(m) = self.at(s, start, b)? {
                return Ok(Some(m));
            }
        }
        Ok(None)
    }

    pub fn search(&self, s: &[char], b: &Budget) -> Result<Option<Match>, OverBudget> {
        self.search_from(s, 0, b)
    }

    /// `pattern.match(s)`
    pub fn match_at_start(&self, s: &[char], b: &Budget) -> Result<Option<Match>, OverBudget> {
        self.at(s, 0, b)
    }

    /// `pattern.finditer(s)` (the frozen patterns never match the empty string; an empty match advances by one)
    pub fn find_all(&self, s: &[char], b: &Budget) -> Result<Vec<Match>, OverBudget> {
        let mut out = Vec::new();
        let mut pos = 0;
        while pos <= s.len() {
            let Some(m) = self.search_from(s, pos, b)? else {
                break;
            };
            pos = if m.end > m.start { m.end } else { m.end + 1 };
            out.push(m);
        }
        Ok(out)
    }

    /// `pattern.sub(replacement, s)` with a function of the match text
    pub fn sub_with(
        &self,
        s: &[char],
        b: &Budget,
        mut f: impl FnMut(&[char]) -> String,
    ) -> Result<Vec<char>, OverBudget> {
        let mut out = Vec::with_capacity(s.len());
        let mut pos = 0;
        for m in self.find_all(s, b)? {
            out.extend_from_slice(&s[pos..m.start]);
            out.extend(f(&s[m.start..m.end]).chars());
            pos = m.end;
        }
        out.extend_from_slice(&s[pos..]);
        Ok(out)
    }

    /// `pattern.sub(replacement, s)` with a literal replacement
    pub fn sub(&self, s: &[char], rep: &str, b: &Budget) -> Result<Vec<char>, OverBudget> {
        self.sub_with(s, b, |_| rep.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cs(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    fn all(p: &str, s: &str) -> Vec<(usize, usize)> {
        let b = Budget::new(DEFAULT_BUDGET);
        Regex::new(p, false)
            .find_all(&cs(s), &b)
            .unwrap()
            .iter()
            .map(|m| (m.start, m.end))
            .collect()
    }

    #[test]
    fn lazy_and_greedy_follow_backtracking_order() {
        assert_eq!(all("a+?", "aaa"), vec![(0, 1), (1, 2), (2, 3)]);
        assert_eq!(all("a+", "aaa"), vec![(0, 3)]);
        assert_eq!(all("a|ab", "ab"), vec![(0, 1)]);
        assert_eq!(all(r"[^,]{0,2}?b", "aab"), vec![(0, 3)]);
    }

    #[test]
    fn lookaround_and_boundaries() {
        assert_eq!(
            all(r"(?<![\w])no(?![\w])", "no nor no"),
            vec![(0, 2), (7, 9)]
        );
        assert_eq!(all(r"\bcat\b", "cat concat cat_"), vec![(0, 3)]);
        assert_eq!(all(r"(?<=\S)'s", "x's 's"), vec![(1, 3)]);
    }

    #[test]
    fn end_allows_a_final_newline() {
        assert_eq!(all(r"a$", "a\n"), vec![(0, 1)]);
        assert_eq!(all(r"a$", "a\n\n"), vec![]);
    }

    #[test]
    fn ignorecase_uses_the_python_equivalences() {
        let b = Budget::new(1000);
        let r = Regex::new("kos", true);
        assert!(r.search(&cs("\u{212A}O\u{17F}"), &b).unwrap().is_some()); // Kelvin sign and long s
    }

    #[test]
    fn the_budget_stops_a_blowup() {
        let b = Budget::new(10_000);
        let r = Regex::new(r":\s*.+\s+or\s+.+\?*$", true);
        let text = cs(&format!(": {}!", "a or ".repeat(2000)));
        assert_eq!(r.search(&text, &b), Err(OverBudget));
    }

    #[test]
    fn groups() {
        let b = Budget::new(1000);
        let m = Regex::new("§([01])§", false)
            .search(&cs("x §1§"), &b)
            .unwrap()
            .unwrap();
        assert_eq!(m.groups, vec![Some((3, 4))]);
    }
}
