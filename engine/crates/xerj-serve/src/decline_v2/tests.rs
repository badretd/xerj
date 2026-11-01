//! decline_v1/tests/test_rules.py and test_roles.py, case for case (synthetic sentences and rows, no bench question). Not ported:
//! test_rules.py `Guard` (the Python check-half guard has no Rust counterpart: the engine reads no split). Added: the policy order,
//! reason codes and the served status, which the Python tests do not cover.

use super::pyre::{Budget, DEFAULT_BUDGET};
use super::replay::swapped_program;
use super::roles as ro;
use super::rules as r;
use super::*;

fn bud() -> Budget {
    Budget::new(DEFAULT_BUDGET)
}

/// (op, result, optional CHAIN fact id)
type S<'a> = (&'a str, &'a str, Option<&'a str>);

fn row(steps: &[S], answer: Option<&str>, evidence: &[String], status: &str) -> Row {
    let mut program = Vec::new();
    let mut trace = Vec::new();
    for s in steps {
        program.push(Step {
            op: s.0.into(),
            pointer: Some(0),
        });
        trace.push(TraceStep {
            op: s.0.into(),
            pointer: Some(0),
            result: Some(s.1.to_string()),
            hop_candidate_ids: s.2.map(|f| vec![f.to_string()]),
        });
    }
    program.push(Step {
        op: "RETURN".into(),
        pointer: None,
    });
    trace.push(TraceStep {
        op: "RETURN".into(),
        pointer: None,
        result: None,
        hop_candidate_ids: None,
    });
    Row {
        question: String::new(),
        status: status.into(),
        answer: answer.map(str::to_string),
        program,
        trace,
        evidence_ids: evidence.to_vec(),
    }
}

fn fact(rel: &str) -> String {
    format!("w1/d1@v1/{rel}")
}

fn fact_d(rel: &str, doc: &str) -> String {
    format!("w1/{doc}@v1/{rel}")
}

fn look_num(v: &str, rel: &str) -> Row {
    let res = format!("VALUE number {v}");
    row(&[("LOOKUP", &res, None)], Some(v), &[fact(rel)], "ANSWER")
}

fn look_txt(v: &str, rel: &str) -> Row {
    let res = format!("VALUE text {v}");
    row(&[("LOOKUP", &res, None)], Some(v), &[fact(rel)], "ANSWER")
}

fn cmp(op: &str, res: &str, ans: &str, rel: &str) -> Row {
    row(
        &[
            ("LOOKUP", "VALUE number 5", None),
            ("LOOKUP", "VALUE number 7", None),
            (op, res, None),
        ],
        Some(ans),
        &[fact_d(rel, "d1"), fact_d(rel, "d2")],
        "ANSWER",
    )
}

fn eq_not_yes(rel: &str) -> Row {
    row(
        &[
            ("LOOKUP", "VALUE text A1", None),
            ("LOOKUP", "VALUE text B2", None),
            ("EQ", "FALSE", None),
            ("NOT", "TRUE", None),
        ],
        Some("yes"),
        &[fact_d(rel, "d1"), fact_d(rel, "d2")],
        "ANSWER",
    )
}

fn date_row(v: &str, rel: &str) -> Row {
    let res = format!("VALUE date {v}");
    row(&[("LOOKUP", &res, None)], Some(v), &[fact(rel)], "ANSWER")
}

fn code(q: &str, x: &Row) -> Option<&'static str> {
    r::check(q, x, &bud()).map(|d| d.code)
}

fn toks(q: &str) -> Vec<String> {
    let b = bud();
    r::tokenize(&r::mask(q, &b).unwrap(), &b).unwrap()
}

// ---- test_rules.py Masking
#[test]
fn titles_parentheses_values_possessives() {
    let t = toks("Is the work 'Children's Games' (object ID 5)'s width greater than that of the work \"Rock and Roll\" (object ID 77)?");
    assert_eq!(t.iter().filter(|x| *x == "QTITLE").count(), 2);
    assert_eq!(t.iter().filter(|x| *x == "PAREN").count(), 2);
    assert!(t.iter().any(|x| x == "POSS"));
    assert!(!t.iter().any(|x| x == "and"));
}

#[test]
fn codes_and_numbers() {
    let t = toks("For customer ID 12531 at airport YKYN on 2011-08-18, return the country.");
    for w in ["VALTOK", "CODETOK", "ID"] {
        assert!(t.iter().any(|x| x == w), "{w}");
    }
}

// ---- RuleA
#[test]
fn coordinated_attributes() {
    assert_eq!(
        code(
            "What are the height and width of the painting 'Blue Sky'?",
            &look_num("12", "height")
        ),
        Some("D2A_MULTI_VALUE")
    );
}

#[test]
fn comma_list() {
    let x = row(
        &[
            ("LOOKUP", "VALUE text a1", None),
            ("CHAIN", "VALUE text 96", Some("w1/d2@v1/architect-id")),
        ],
        Some("96"),
        &[fact("architect"), fact_d("architect-id", "d2")],
        "ANSWER",
    );
    assert_eq!(
        code(
            "Give the name, ID and birth year of the architect of the Tower.",
            &x
        ),
        Some("D2A_MULTI_VALUE")
    );
}

#[test]
fn together_with_and_pair() {
    assert_eq!(
        code(
            "Report the depth together with the place of event Q12.",
            &look_num("10", "depth")
        ),
        Some("D2A_MULTI_VALUE")
    );
    assert_eq!(
        code(
            "Report the radius of planet Zeta b and its limit flag as a pair.",
            &look_txt("0", "limit-flag")
        ),
        Some("D2A_MULTI_VALUE")
    );
}

#[test]
fn two_word_unit_does_not_hide_the_list() {
    assert_eq!(
        code(
            "What radius in Earth radii and limit flag are recorded for planet Zeta b?",
            &look_num("3.5", "radius")
        ),
        Some("D2A_MULTI_VALUE")
    );
}

#[test]
fn second_question() {
    assert_eq!(
        code(
            "Which architect designed the Tower, and what nationality does the architect have?",
            &look_txt("Dutch", "nationality")
        ),
        Some("D2A_MULTI_VALUE")
    );
}

#[test]
fn list_request() {
    assert_eq!(
        code("List the moons of Mars.", &look_txt("Phobos", "moon")),
        Some("D2A_MULTI_VALUE")
    );
}

#[test]
fn entity_names_with_and_are_not_lists() {
    assert_eq!(
        code(
            "What is the population of Bosnia and Herzegovina?",
            &look_num("3200000", "population")
        ),
        None
    );
    assert_eq!(
        code(
            "Is the height of 'Rock and Roll' greater than that of 'Jazz'?",
            &cmp("GT", "FALSE", "no", "height")
        ),
        None
    );
}

#[test]
fn plural_noun_alone_is_not_a_trigger() {
    assert_eq!(
        code(
            "Which customer ID values are recorded on invoice 5?",
            &look_txt("17389", "customer-id")
        ),
        None
    );
}

// ---- RuleB
#[test]
fn both() {
    assert_eq!(
        code(
            "Which surface types occur at both Alpha Airport and Beta Airport?",
            &eq_not_yes("surface")
        ),
        Some("D2B_SET")
    );
}

#[test]
fn intersection() {
    assert_eq!(
        code(
            "Find the intersection of the product sets of customers 1 and 2.",
            &look_txt("invoice 9", "invoice")
        ),
        Some("D2B_SET")
    );
}

#[test]
fn existential() {
    assert_eq!(
        code(
            "Is there a lighted runway at Gamma Field?",
            &eq_not_yes("lighted")
        ),
        Some("D2B_EXISTENTIAL")
    );
    assert_eq!(
        code(
            "Does any painting by Ana Ruiz hang in the department 'Film'?",
            &eq_not_yes("department")
        ),
        Some("D2B_EXISTENTIAL")
    );
}

#[test]
fn singular_unique_and_noun_share_are_not_sets() {
    assert_eq!(
        code(
            "What is the unique identifier of the station Delta?",
            &look_txt("ST-9", "identifier")
        ),
        None
    );
    assert_eq!(
        code(
            "What share of the votes did the Green Party win?",
            &look_num("12.5", "vote-share")
        ),
        None
    );
}

// ---- RuleC
#[test]
fn count() {
    assert_eq!(
        code(
            "How many runways does Kappa Airport have?",
            &look_num("3", "runway")
        ),
        Some("D2C_AGGREGATE")
    );
}

#[test]
fn average_unless_the_field_is_the_average() {
    assert_eq!(
        code(
            "What is the average depth of Lake Tahoe?",
            &look_num("300", "depth")
        ),
        Some("D2C_AGGREGATE")
    );
    assert_eq!(
        code(
            "What is the average depth of Lake Tahoe?",
            &look_num("300", "average-depth")
        ),
        None
    );
}

#[test]
fn maximum_unless_the_field_is_the_maximum() {
    assert_eq!(
        code(
            "What is the maximum speed of the Model T?",
            &look_num("72", "speed")
        ),
        Some("D2C_EXTREMUM")
    );
    assert_eq!(
        code(
            "What is the maximum speed of the Model T?",
            &look_num("72", "max-speed")
        ),
        None
    );
}

#[test]
fn superlative_over_two_named_options() {
    let pick = row(
        &[
            ("LOOKUP", "VALUE number 6", None),
            ("LOOKUP", "VALUE number 7", None),
            ("PICK_GT", "VALUE text Venus", None),
        ],
        Some("Venus"),
        &[fact_d("radius", "d1"), fact_d("radius", "d2")],
        "ANSWER",
    );
    assert_eq!(code("Which is the largest: Mars or Venus?", &pick), None);
    assert_eq!(
        code(
            "Which is the largest: Mars or Venus?",
            &look_txt("Venus", "name")
        ),
        Some("D2C_EXTREMUM")
    );
    let pick2 = row(
        &[
            ("LOOKUP", "VALUE number 6650", None),
            ("LOOKUP", "VALUE number 6400", None),
            ("PICK_GT", "VALUE text Nile", None),
        ],
        Some("Nile"),
        &[fact_d("length", "d1"), fact_d("length", "d2")],
        "ANSWER",
    );
    assert_eq!(
        code("Which is the longest, the Nile or the Amazon?", &pick2),
        None
    );
}

#[test]
fn measurement_is_not_a_count() {
    assert_eq!(
        code(
            "How many days is the orbital period of Mars?",
            &look_num("687", "orbital-period")
        ),
        None
    );
}

#[test]
fn count_comparison() {
    assert_eq!(
        code(
            "Does customer 7 have more invoices than customer 9?",
            &cmp("GT", "TRUE", "yes", "invoice-count")
        ),
        Some("D2C_COUNT_COMPARISON")
    );
}

#[test]
fn comparative_without_a_comparison() {
    assert_eq!(
        code(
            "Which Scottish airport is bigger?",
            &look_txt("Heathrow", "name")
        ),
        Some("D2C_COMPARATIVE_NO_COMPARISON")
    );
}

#[test]
fn comparative_with_equality_only() {
    assert_eq!(
        code("Is the Nile longer than the Amazon?", &eq_not_yes("length")),
        Some("D2C_COMPARATIVE_EQ_ONLY")
    );
    assert_eq!(
        code(
            "Is the Nile longer than the Amazon?",
            &cmp("GT", "TRUE", "yes", "length")
        ),
        None
    );
}

#[test]
fn arithmetic_and_first_last() {
    let d = r::check(
        "Subtract the first invoice date from the last for customer 4.",
        &date_row("2011-01-02", "invoice-date"),
        &bud(),
    );
    assert_eq!(d.unwrap().rule, "c");
}

#[test]
fn group() {
    assert_eq!(
        r::check(
            "Count the stations per region.",
            &look_num("4", "station"),
            &bud()
        )
        .unwrap()
        .rule,
        "c"
    );
}

#[test]
fn not_extremum() {
    assert_eq!(
        code(
            "What first-level administrative region is recorded for Omega Airport?",
            &look_txt("Kent", "region")
        ),
        None
    );
    assert_eq!(
        code(
            "What is the last name of the artist Anna Berg?",
            &look_txt("Berg", "last-name")
        ),
        None
    );
    assert_eq!(
        code(
            "Which country contains the Max Planck Institute?",
            &look_txt("Germany", "country")
        ),
        None
    );
    assert_eq!(
        code(
            "Is runway Echo at least 3000 feet long?",
            &cmp("GT", "TRUE", "yes", "length")
        ),
        None
    );
}

#[test]
fn before_comparison() {
    let x = row(
        &[
            ("LOOKUP", "VALUE date 1990-01-01", None),
            ("LOOKUP", "VALUE date 1991-01-01", None),
            ("LT", "TRUE", None),
        ],
        Some("yes"),
        &[
            fact_d("acquisition-date", "d1"),
            fact_d("acquisition-date", "d2"),
        ],
        "ANSWER",
    );
    assert_eq!(
        code("Was the work 'X' acquired before the work 'Y'?", &x),
        None
    );
}

// ---- RuleD
#[test]
fn action_request() {
    assert_eq!(
        code(
            "Refund invoice 412233 and email the customer.",
            &look_txt("17850", "customer-id")
        ),
        Some("D2D_ACTION_REQUEST")
    );
    assert_eq!(
        code(
            "Reserve The Night Watch for a loan next June.",
            &date_row("1941-06-10", "acquisition-date")
        ),
        Some("D2D_ACTION_REQUEST")
    );
}

#[test]
fn keyword_query_is_not_an_action() {
    assert_eq!(
        code(
            "Email address of Jo Lee",
            &look_txt("jo@example.org", "email-address")
        ),
        None
    );
}

#[test]
fn yes_no_for_wh() {
    assert_eq!(
        code(
            "Which country contains Sigma Airport?",
            &eq_not_yes("country")
        ),
        Some("D2D_YES_NO_FOR_WH")
    );
}

#[test]
fn value_for_yes_no() {
    assert_eq!(
        code(
            "Can a large jet land at Tau Airport tonight?",
            &look_txt("Tau Town", "municipality")
        ),
        Some("D2D_VALUE_FOR_YES_NO")
    );
}

#[test]
fn indirect_requests() {
    assert_eq!(
        code(
            "Can you tell me the country of Tau Airport?",
            &look_txt("Nepal", "country")
        ),
        None
    );
    assert_eq!(
        code(
            "Tell me whether the Nile is longer than the Amazon.",
            &cmp("GT", "TRUE", "yes", "length")
        ),
        None
    );
    assert_eq!(
        code(
            "Do you know the country of Tau Airport?",
            &look_txt("Nepal", "country")
        ),
        None
    );
    assert_eq!(
        code(
            "Can you tell me whether the Nile is longer than the Amazon?",
            &cmp("GT", "TRUE", "yes", "length")
        ),
        None
    );
}

#[test]
fn identifier_not_requested() {
    let x = row(
        &[
            ("LOOKUP", "VALUE text Nepal", None),
            ("CHAIN", "VALUE text NP", Some("w1/d2@v1/country-code")),
        ],
        Some("NP"),
        &[fact("country"), fact_d("country-code", "d2")],
        "ANSWER",
    );
    assert_eq!(
        code("Which country contains Upsilon Airport?", &x),
        Some("D2D_ID_NOT_REQUESTED")
    );
    let q = "Is Upsilon Airport's runway longer than Phi Airport's?";
    assert_eq!(
        code(q, &eq_not_yes("identifier")),
        Some("D2C_COMPARATIVE_EQ_ONLY")
    );
    let all = r::check_all(q, &eq_not_yes("identifier"), &bud()).unwrap();
    assert!(all.iter().any(|d| d.code == "D2D_ID_NOT_REQUESTED"));
}

#[test]
fn identifier_requested() {
    assert_eq!(
        code(
            "What is the IATA code of Upsilon Airport?",
            &look_txt("UPS", "iata-code")
        ),
        None
    );
    assert_eq!(
        code(
            "Which customer ID is recorded on invoice 12?",
            &look_txt("invoice 12 line 1", "invoice")
        ),
        Some("D2D_ID_REQUESTED_NOT_ID")
    );
}

#[test]
fn when() {
    let q = "When was the bridge 'Golden Arc' opened?";
    assert_eq!(
        code(q, &look_txt("Golden City", "city")),
        Some("D2D_TYPE_DATE")
    );
    assert_eq!(code(q, &date_row("1937-05-27", "opening-date")), None);
    assert_eq!(code(q, &look_num("1937", "opening-year")), None);
}

#[test]
fn how_tall() {
    assert_eq!(
        code(
            "How tall is the tower 'Spire'?",
            &look_txt("steel", "material")
        ),
        Some("D2D_TYPE_NUMBER")
    );
    assert_eq!(
        code("How tall is the tower 'Spire'?", &look_num("324", "height")),
        None
    );
}

#[test]
fn who_where() {
    assert_eq!(
        code("Who designed the tower 'Spire'?", &look_num("1889", "year")),
        Some("D2D_TYPE_PERSON")
    );
    assert_eq!(
        code(
            "Who designed the tower 'Spire'?",
            &look_txt("Gustave Eiffel", "designer")
        ),
        None
    );
    assert_eq!(
        code(
            "Where is the tower 'Spire'?",
            &date_row("1889-03-31", "opening-date")
        ),
        Some("D2D_TYPE_PLACE")
    );
}

#[test]
fn place_granularity() {
    assert_eq!(
        code(
            "In which country is Phi Airport listed?",
            &look_txt("Bagmati", "region")
        ),
        Some("D2D_PLACE_FIELD_MISMATCH")
    );
    assert_eq!(
        code(
            "In which country is Phi Airport listed?",
            &look_txt("Nepal", "country")
        ),
        None
    );
    assert_eq!(
        code(
            "Locate the administrative region containing Phi Airport.",
            &look_txt("Bagmati", "region")
        ),
        None
    );
    assert_eq!(
        code(
            "Which state is the city of Austin in?",
            &look_txt("Texas", "region")
        ),
        None
    );
}

#[test]
fn rates_and_times() {
    assert_eq!(
        code(
            "What is the income per capita of Chile?",
            &look_num("16000", "income-per-capita")
        ),
        None
    );
    assert_eq!(
        code(
            "What time does the museum 'Prado' open?",
            &look_txt("9:00", "opening-time")
        ),
        None
    );
}

#[test]
fn only_accepted_answers_are_checked() {
    let x = row(&[("LOOKUP", "UNKNOWN", None)], None, &[], "UNKNOWN");
    assert_eq!(code("How many runways does Kappa Airport have?", &x), None);
}

// ---- test_roles.py
fn roles(q: &str, names: [&str; 2], kind: Option<&str>) -> ro::Roles {
    ro::read_roles(q, names, [kind, kind], &bud()).unwrap()
}

fn read3(q: &str, kind: Option<&str>) -> (usize, usize, &'static str) {
    match roles(q, ["Alpha", "Beta"], kind) {
        ro::Roles::Read {
            subject,
            reference,
            expected,
            ..
        } => (subject, reference, expected),
        o => panic!("{o:?}"),
    }
}

fn undet(q: &str, kind: Option<&str>) -> bool {
    matches!(
        roles(q, ["Alpha", "Beta"], kind),
        ro::Roles::Undetermined(_)
    )
}

const NUM: Option<&str> = Some("number");
const DATE: Option<&str> = Some("date");

#[test]
fn than() {
    assert_eq!(
        read3("Is Alpha's height greater than Beta's?", NUM),
        (0, 1, "GT")
    );
    assert_eq!(
        read3(
            "Is the recorded width of Beta smaller than that of Alpha?",
            NUM
        ),
        (1, 0, "LT")
    );
}

#[test]
fn compared_with_and_reference() {
    assert_eq!(
        read3("Compared with Beta, is Alpha taller?", NUM),
        (0, 1, "GT")
    );
    assert_eq!(
        read3(
            "Using Beta as the reference, is Alpha's width smaller?",
            NUM
        ),
        (0, 1, "LT")
    );
    assert_eq!(
        read3(
            "With Alpha as the baseline, does Beta have a longer period?",
            NUM
        ),
        (1, 0, "GT")
    );
}

#[test]
fn passive_and_temporal() {
    assert_eq!(
        read3("Is Alpha's width exceeded by the width of Beta?", NUM),
        (0, 1, "LT")
    );
    assert_eq!(
        read3(
            "Was Alpha acquired before Beta, using their recorded dates?",
            DATE
        ),
        (0, 1, "LT")
    );
    assert_eq!(read3("Was Beta built after Alpha?", DATE), (1, 0, "GT"));
}

#[test]
fn older_depends_on_type() {
    assert_eq!(read3("Is Alpha older than Beta?", DATE).2, "LT");
    assert_eq!(read3("Is Alpha older than Beta?", NUM).2, "GT");
    assert!(undet("Is Alpha older than Beta?", None));
}

#[test]
fn left_alone() {
    assert!(undet("Is Alpha not higher than Beta?", NUM));
    assert!(undet("Are Alpha and Beta of equal height?", NUM));
    assert!(undet("Is Alpha higher and lower than Beta?", NUM));
    assert!(undet("Is Gamma higher than Delta?", NUM));
}

#[test]
fn commas_inside_titles_and_names_inside_names() {
    let q = "Is the work 'Lima, Peru (Plan)' (ID 1)'s width greater than that of the work 'Lima, Peru (Study)' (ID 2)?";
    let ro::Roles::Read {
        subject,
        reference,
        expected,
        ..
    } = roles(q, ["Lima, Peru (Plan)", "Lima, Peru (Study)"], NUM)
    else {
        panic!()
    };
    assert_eq!((subject, reference, expected), (0, 1, "GT"));
    let q = "Compared with the work 'Seattle, Park' (object ID 7), is the work 'Seattle, Park, North' (object ID 8) wider?";
    let ro::Roles::Read {
        subject, reference, ..
    } = roles(q, ["object ID 8", "object ID 7"], NUM)
    else {
        panic!()
    };
    assert_eq!((subject, reference), (0, 1));
    let ro::Roles::Read {
        subject,
        reference,
        expected,
        ..
    } = roles(
        "Compared with Olympic Park, is Olympic Park North wider?",
        ["Olympic Park North", "Olympic Park"],
        None,
    )
    else {
        panic!()
    };
    assert_eq!((subject, reference, expected), (0, 1, "GT"));
}

fn prog(ops: &[&str]) -> Row {
    let mut program: Vec<Step> = ops
        .iter()
        .map(|o| Step {
            op: o.to_string(),
            pointer: (*o == "LOOKUP").then_some(0),
        })
        .collect();
    program.push(Step {
        op: "RETURN".into(),
        pointer: None,
    });
    Row {
        program,
        ..Row::default()
    }
}

fn reads(s1: &str, s2: &str, rel1: &str, rel2: &str) -> Vec<Read> {
    let one = |t: usize, s: &str, rel: &str, v: &str| Read {
        step: t,
        op: "LOOKUP".into(),
        subject: Some(s.to_string()),
        relation: Some(rel.to_string()),
        fact_id: Some(format!("w/d{t}@v1/{rel}")),
        result: Some(format!("VALUE number {v}")),
    };
    vec![one(0, s1, rel1, "5"), one(1, s2, rel2, "7")]
}

const Q: &str = "Is Alpha's height greater than Beta's?";

fn role_code(ops: &[&str], rd: Vec<Read>) -> ro::RoleVerdict {
    ro::check(Q, &prog(ops), &rd, &bud()).unwrap()
}

const CMP_GT: [&str; 3] = ["LOOKUP", "LOOKUP", "GT"];
const CMP_LT: [&str; 3] = ["LOOKUP", "LOOKUP", "LT"];
const NOT_GT: [&str; 4] = ["LOOKUP", "LOOKUP", "GT", "NOT"];
const NOT_LT: [&str; 4] = ["LOOKUP", "LOOKUP", "LT", "NOT"];

#[test]
fn consistent() {
    assert_eq!(
        role_code(&CMP_GT, reads("Alpha", "Beta", "height", "height")).code,
        None
    );
    assert_eq!(
        role_code(&CMP_LT, reads("Beta", "Alpha", "height", "height")).code,
        None
    );
}

#[test]
fn reversed_is_repairable() {
    let v = role_code(&CMP_GT, reads("Beta", "Alpha", "height", "height"));
    assert_eq!(v.code, Some("D2R_DIRECTION_REVERSED"));
    assert!(v.repairable);
    assert_eq!(
        role_code(&CMP_LT, reads("Alpha", "Beta", "height", "height")).code,
        Some("D2R_DIRECTION_REVERSED")
    );
}

#[test]
fn negated() {
    let v = role_code(&NOT_GT, reads("Alpha", "Beta", "height", "height"));
    assert_eq!(v.code, Some("D2R_DIRECTION_NEGATED"));
    assert!(!v.repairable);
    assert_eq!(
        role_code(&NOT_LT, reads("Beta", "Alpha", "height", "height")).code,
        Some("D2R_DIRECTION_NEGATED")
    );
    assert_eq!(
        role_code(&NOT_LT, reads("Alpha", "Beta", "height", "height")).code,
        None
    );
}

#[test]
fn operand_faults() {
    assert_eq!(
        role_code(&CMP_GT, reads("Alpha", "Alpha", "height", "height")).code,
        Some("D2R_SAME_OPERAND")
    );
    assert_eq!(
        role_code(&CMP_GT, reads("Alpha", "Beta", "height", "width")).code,
        Some("D2R_RELATION_MISMATCH")
    );
    assert_eq!(
        role_code(&CMP_GT, reads("Alpha", "Gamma", "height", "height")).code,
        Some("D2R_OPERAND_NOT_NAMED")
    );
}

#[test]
fn other_programs_are_not_checked() {
    for ops in [
        &["LOOKUP", "LOOKUP", "PICK_GT"][..],
        &["LOOKUP", "LOOKUP", "EQ"],
        &["LOOKUP", "CHAIN", "LOOKUP", "GT"],
        &["LOOKUP"],
    ] {
        let v = role_code(ops, reads("Beta", "Alpha", "height", "height"));
        assert_eq!(v.code, None);
        assert!(!v.applies);
    }
}

#[test]
fn swapped_program_shapes() {
    let mut p = prog(&CMP_GT).program;
    p[0].pointer = Some(3);
    p[1].pointer = Some(7);
    let sw = swapped_program(&p).unwrap();
    let got: Vec<(&str, Option<i64>)> = sw.iter().map(|s| (s.op.as_str(), s.pointer)).collect();
    assert_eq!(
        got,
        vec![
            ("LOOKUP", Some(7)),
            ("LOOKUP", Some(3)),
            ("GT", None),
            ("RETURN", None)
        ]
    );
    assert!(swapped_program(&prog(&["LOOKUP", "CHAIN", "LOOKUP", "GT"]).program).is_none());
    assert!(swapped_program(&prog(&["LOOKUP", "LOOKUP", "EQ"]).program).is_none());
}

// ---- added: the policy and the served outcome
struct Stub {
    reads: Vec<Read>,
    swapped: Option<Swapped>,
    matches: bool,
}

impl Replay for Stub {
    fn matches(&mut self, _r: &Row) -> bool {
        self.matches
    }
    fn operand_reads(&mut self, _r: &Row) -> Vec<Read> {
        self.reads.clone()
    }
    fn execute_swapped(&mut self, _r: &Row) -> Option<Swapped> {
        self.swapped.clone()
    }
}

fn stub() -> Stub {
    Stub {
        reads: Vec::new(),
        swapped: None,
        matches: true,
    }
}

fn with_q(q: &str, mut x: Row) -> Row {
    x.question = q.to_string();
    x
}

fn cmp_row(q: &str, ans: &str) -> Row {
    with_q(q, cmp("GT", "TRUE", ans, "height"))
}

#[test]
fn an_answer_that_passes_is_served_with_no_reason() {
    let d = decide(
        &with_q("What is the height of Alpha?", look_num("5", "height")),
        &mut stub(),
    );
    assert!(d.accepted);
    assert_eq!(d.reason_code, None);
    assert_eq!(d.served("ANSWER"), ("ANSWER", Some("5".to_string())));
}

#[test]
fn counting_is_declined_whatever_the_model_did() {
    for raw in ["ANSWER", "UNKNOWN", "CONFLICT"] {
        let mut x = with_q(
            "How many runways does Kappa Airport have?",
            look_num("3", "runway"),
        );
        x.status = raw.to_string();
        let d = decide(&x, &mut stub());
        assert!(!d.accepted);
        assert_eq!(d.reason_code.as_deref(), Some("COUNTING_OFF"), "{raw}");
        assert_eq!(d.policy_code.as_deref(), Some("P3"));
        assert_eq!(d.served(raw), ("UNSUPPORTED", None));
    }
}

#[test]
fn counting_wording_past_the_scope_guard_is_named_too() {
    let d = decide(
        &with_q(
            "Report the tally of stations in Kent.",
            look_num("3", "station"),
        ),
        &mut stub(),
    );
    assert_eq!(d.policy_code.as_deref(), Some("D2C_AGGREGATE"));
    assert_eq!(d.reason_code.as_deref(), Some("COUNTING_OFF"));
}

#[test]
fn a_counting_question_refused_by_another_rule_still_names_counting() {
    let d = decide(
        &with_q(
            "Report the tally of Spire stations.",
            look_txt("Spire", "name"),
        ),
        &mut stub(),
    );
    assert!(!d.accepted);
    assert_eq!(d.policy_code.as_deref(), Some("P3"));
    assert_eq!(d.detail, "copy guard: the value is text of the question");
    assert_eq!(d.reason_code.as_deref(), Some("COUNTING_OFF"));
    assert_eq!(d.served("ANSWER"), ("UNSUPPORTED", None));
}

#[test]
fn every_decline_has_a_reason() {
    let scope = decide(
        &with_q("Which is the tallest tower?", look_txt("Spire", "name")),
        &mut stub(),
    );
    assert_eq!(scope.reason_code.as_deref(), Some("P3_SCOPE"));
    let copy = decide(
        &with_q("What is the name of Spire?", look_txt("Spire", "name")),
        &mut stub(),
    );
    assert_eq!(copy.reason_code.as_deref(), Some("P3_COPY"));
    let d2 = decide(
        &with_q(
            "Which Scottish airport is bigger?",
            look_txt("Heathrow", "name"),
        ),
        &mut stub(),
    );
    assert_eq!(
        d2.reason_code.as_deref(),
        Some("D2C_COMPARATIVE_NO_COMPARISON")
    );
    assert_eq!(d2.served("ANSWER"), ("UNSUPPORTED", None));
    let mut raw = with_q(
        "What is the height of Alpha?",
        row(&[("LOOKUP", "UNKNOWN", None)], None, &[], "UNKNOWN"),
    );
    raw.status = "UNKNOWN".to_string();
    let d = decide(&raw, &mut stub());
    assert_eq!(d.reason_code.as_deref(), Some("RAW_UNKNOWN"));
    assert_eq!(d.served("UNKNOWN"), ("UNKNOWN", None));
}

#[test]
fn a_reversed_comparison_is_repaired_and_logged() {
    let mut s = stub();
    s.reads = reads("Beta", "Alpha", "height", "height");
    s.swapped = Some(Swapped {
        status: "ANSWER".into(),
        answer: Some("no".into()),
        evidence_ids: vec!["w/d0@v1/height".into()],
        program: prog(&CMP_GT).program,
    });
    let d = decide(&cmp_row(Q, "yes"), &mut s);
    assert!(d.accepted);
    assert_eq!(d.reason_code.as_deref(), Some("D2R_REPAIRED"));
    assert_eq!(d.served("ANSWER"), ("ANSWER", Some("no".to_string())));
    let j = d.to_json("ANSWER", &[]);
    assert_eq!(j["repair"]["original_answer"], "yes");
    assert_eq!(j["repair"]["repaired_answer"], "no");
    assert_eq!(j["final_evidence_ids"][0], "w/d0@v1/height");
}

#[test]
fn a_repair_without_an_answer_declines() {
    let mut s = stub();
    s.reads = reads("Beta", "Alpha", "height", "height");
    s.swapped = Some(Swapped {
        status: "UNKNOWN".into(),
        answer: None,
        evidence_ids: Vec::new(),
        program: Vec::new(),
    });
    let d = decide(&cmp_row(Q, "yes"), &mut s);
    assert!(!d.accepted);
    assert_eq!(d.reason_code.as_deref(), Some("D2R_REPAIR_NOT_ANSWER"));
}

#[test]
fn every_error_path_fails_closed() {
    let mut s = stub();
    s.matches = false;
    let d = decide(&cmp_row(Q, "yes"), &mut s);
    assert!(!d.accepted);
    assert_eq!(d.reason_code.as_deref(), Some("D2R_REPLAY_MISMATCH"));
    // the rule budget is spent: declined, not answered
    let mut s = stub();
    let d = decide_budget(
        &with_q("What is the height of Alpha?", look_num("5", "height")),
        &mut s,
        3,
    );
    assert!(!d.accepted);
    assert_eq!(d.reason_code.as_deref(), Some("D2X_RULE_ERROR"));
    assert_eq!(d.served("ANSWER"), ("UNSUPPORTED", None));
}

#[test]
fn python_string_details() {
    use super::pystr::{lower, upper};
    assert_eq!(lower("ΣΑΣ"), "σας");
    assert_eq!(lower("Σ"), "σ");
    assert_eq!(lower("İ"), "i\u{307}");
    assert_eq!(upper("ß"), "SS");
    assert_eq!(upper("ıd"), "ID");
}
