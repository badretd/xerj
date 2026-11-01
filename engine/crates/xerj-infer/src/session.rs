//! One question's inference state with exact prefix reuse across its decisions.
//!
//! What is shared between the decisions of one question (inputs.py / stack_v2.py): decision t's tokens are
//! `P + H_0 .. H_{t-1} + suffix_t`, where P (question, external_executor: candidate lines, "Steps:") is fixed, each executed
//! action appends one history segment H, and the suffix ("\nNext:" or a stack-v2 / v5 hop block) belongs to one decision only.
//!
//! * **external_executor** (plain causal decoder + a position-wise capacity block): every layer's K/V of P + H is exact for all
//!   later decisions. P runs once per question (the base candidate lines' pointer keys come from it); each decision appends
//!   the new history segment to the cache (K/V only at the last layer) and runs only its suffix on top.
//! * **rrt**: the conditioner adds register attention to EVERY position after every layer, and the registers change after
//!   each action, so layers 1..6 must be recomputed for all positions. Exactly shared and cached: the embeddings, layer 0's K/V
//!   and layer 0's output before the hook for P + H. The memory tokens' K/V (one shared conditioner = the same K/V at every
//!   layer) are computed once per question (stack-v2: the hop half again when the hop manifest changes). The last layer runs
//!   attention / MLP / hook for the last position only.

use crate::codec::{self, StackEntry, REGISTER_COUNT, REGISTER_WIDTH};
use crate::config::{Arm, StackProtocol};
use crate::error::{Error, Result};
use crate::inputs::{self, CandidateFields};
use crate::kernels::ops;
use crate::model::{grow, Buffers, CrossKv, LayerKv, Model};
use crate::tokenizer::TextEncoder;
use std::collections::HashMap;

/// the hop manifest a stack-v2 decision reads
#[derive(Debug, Clone, Copy)]
pub struct HopInput<'a> {
    pub cands: &'a [CandidateFields],
    pub resolves: &'a [bool],
    /// identity of the manifest (memory is re-encoded only when it changes)
    pub key: u64,
}

/// outputs of one decision
#[derive(Debug, Clone, Default)]
pub struct DecisionOut {
    pub op_logits: Vec<f32>,
    /// pointer logits in the layout choose_pointer reads: rrt [R+1] (R = K or 2K), external_executor v1 [n_base+1],
    /// external_executor v2 [2K+1]; f32::MIN = masked (torch.finfo(float32).min)
    pub ptr_logits: Vec<f32>,
    /// tokens processed by the decoder for this decision
    pub computed_tokens: usize,
    /// full decision input length
    pub input_tokens: usize,
}

/// cache of tokenized segments (static texts and repeated candidate lines)
#[derive(Default)]
pub struct SegCache {
    map: HashMap<String, Vec<u32>>,
}

impl SegCache {
    pub fn get(&mut self, tok: &dyn TextEncoder, text: &str) -> Result<&[u32]> {
        let _t = crate::prof::start(crate::prof::Cat::Tokenize);
        if !self.map.contains_key(text) {
            if self.map.len() > 50_000 {
                self.map.clear();
            }
            let ids = tok.encode(text)?;
            self.map.insert(text.to_string(), ids);
        }
        Ok(&self.map[text])
    }
}

pub struct Session<'m> {
    pub m: &'m Model,
    tok: &'m dyn TextEncoder,
    pub segs: SegCache,
    b: Buffers,
    kv: Vec<LayerKv>,
    /// committed token ids (P + history)
    pub ids: Vec<u32>,
    /// ids of the current decision's suffix
    pub suffix: Vec<u32>,
    x: Vec<f32>,
    xc: Vec<f32>,
    // ---- rrt
    out0: Vec<f32>,
    l0_rows: usize,
    mem_values: Vec<f32>,
    mem_mask: Vec<bool>,
    mem_tokens: Vec<f32>,
    mem_kv: CrossKv,
    ptr_keys: Vec<f32>,
    rec_valid: Vec<bool>,
    hop_key: Option<u64>,
    reg_tokens: Vec<f32>,
    reg_kv: CrossKv,
    has_regs: bool,
    // ---- external executor
    committed: usize,
    base_keys: Vec<f32>,
    n_base: usize,
    need: Vec<usize>,
    spans_tmp: Vec<(usize, usize)>,
    hop_keys: Vec<f32>,
    q: Vec<f32>,
    lastv: Vec<f32>,
    valid: Vec<bool>,
}

/// The buffers of a finished session without its model borrow: KV caches, activation and scratch buffers, the
/// tokenized-segment cache and the attached pool. A host that answers many questions keeps them and starts the next question
/// with [`Session::from_bufs`], so the KV caches are not allocated and zero-filled again and repeated text segments are not
/// tokenized again. Every per-question state is reset by [`Session::begin`] (the same reuse the parity and bench tools have
/// always done with one session for all questions), so the decisions are identical.
pub struct SessionBufs {
    layers: usize,
    segs: SegCache,
    b: Buffers,
    kv: Vec<LayerKv>,
    ids: Vec<u32>,
    suffix: Vec<u32>,
    x: Vec<f32>,
    xc: Vec<f32>,
    out0: Vec<f32>,
    mem_values: Vec<f32>,
    mem_mask: Vec<bool>,
    mem_tokens: Vec<f32>,
    mem_kv: CrossKv,
    ptr_keys: Vec<f32>,
    rec_valid: Vec<bool>,
    reg_tokens: Vec<f32>,
    reg_kv: CrossKv,
    base_keys: Vec<f32>,
    need: Vec<usize>,
    spans_tmp: Vec<(usize, usize)>,
    hop_keys: Vec<f32>,
    q: Vec<f32>,
    lastv: Vec<f32>,
    valid: Vec<bool>,
}

impl<'m> Session<'m> {
    /// detach the reusable buffers (see [`SessionBufs`])
    pub fn into_bufs(self) -> SessionBufs {
        SessionBufs {
            layers: self.kv.len(),
            segs: self.segs,
            b: self.b,
            kv: self.kv,
            ids: self.ids,
            suffix: self.suffix,
            x: self.x,
            xc: self.xc,
            out0: self.out0,
            mem_values: self.mem_values,
            mem_mask: self.mem_mask,
            mem_tokens: self.mem_tokens,
            mem_kv: self.mem_kv,
            ptr_keys: self.ptr_keys,
            rec_valid: self.rec_valid,
            reg_tokens: self.reg_tokens,
            reg_kv: self.reg_kv,
            base_keys: self.base_keys,
            need: self.need,
            spans_tmp: self.spans_tmp,
            hop_keys: self.hop_keys,
            q: self.q,
            lastv: self.lastv,
            valid: self.valid,
        }
    }

    /// a session over `m` that reuses the buffers of an earlier session of the same model (a new session when the buffers
    /// belong to a model with another layer count)
    pub fn from_bufs(m: &'m Model, tok: &'m dyn TextEncoder, bufs: SessionBufs) -> Self {
        if bufs.layers != m.cfg.layers {
            return Session::new(m, tok);
        }
        let mut s = Session::new(m, tok);
        s.segs = bufs.segs;
        s.b = bufs.b;
        s.b.gemm.force_f32 = false;
        s.b.gemm.calib = None;
        s.kv = bufs.kv;
        s.ids = bufs.ids;
        s.suffix = bufs.suffix;
        s.x = bufs.x;
        s.xc = bufs.xc;
        s.out0 = bufs.out0;
        s.mem_values = bufs.mem_values;
        s.mem_mask = bufs.mem_mask;
        s.mem_tokens = bufs.mem_tokens;
        s.mem_kv = bufs.mem_kv;
        s.ptr_keys = bufs.ptr_keys;
        s.rec_valid = bufs.rec_valid;
        s.reg_tokens = bufs.reg_tokens;
        s.reg_kv = bufs.reg_kv;
        s.base_keys = bufs.base_keys;
        s.need = bufs.need;
        s.spans_tmp = bufs.spans_tmp;
        s.hop_keys = bufs.hop_keys;
        s.q = bufs.q;
        s.lastv = bufs.lastv;
        s.valid = bufs.valid;
        s
    }

    pub fn new(m: &'m Model, tok: &'m dyn TextEncoder) -> Self {
        Session {
            m,
            tok,
            segs: SegCache::default(),
            b: Buffers::default(),
            kv: (0..m.cfg.layers).map(|_| LayerKv::default()).collect(),
            ids: Vec::new(),
            suffix: Vec::new(),
            x: Vec::new(),
            xc: Vec::new(),
            out0: Vec::new(),
            l0_rows: 0,
            mem_values: Vec::new(),
            mem_mask: Vec::new(),
            mem_tokens: Vec::new(),
            mem_kv: CrossKv::default(),
            ptr_keys: Vec::new(),
            rec_valid: Vec::new(),
            hop_key: None,
            reg_tokens: Vec::new(),
            reg_kv: CrossKv::default(),
            has_regs: false,
            committed: 0,
            base_keys: Vec::new(),
            n_base: 0,
            need: Vec::new(),
            spans_tmp: Vec::new(),
            hop_keys: Vec::new(),
            q: Vec::new(),
            lastv: Vec::new(),
            valid: Vec::new(),
        }
    }

    fn seg(&mut self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        let ids = self.segs.get(self.tok, text)?;
        out.extend_from_slice(ids);
        Ok(())
    }

    /// full input ids of the last decision (committed prefix + suffix)
    pub fn input_ids(&self) -> Vec<u32> {
        let mut v = self.ids.clone();
        v.extend_from_slice(&self.suffix);
        v
    }

    /// Start a question: build the shared prefix, encode the memory (rrt) or run the prefix through the decoder
    /// (external_executor). `cands` = the base candidate manifest (<= K records).
    pub fn begin(&mut self, question: &str, cands: &[CandidateFields]) -> Result<()> {
        let m = self.m;
        let cfg = &m.cfg;
        let k = cfg.k;
        if cands.len() > k {
            return Err(Error::Input(format!(
                "{} candidates exceed K = {k}",
                cands.len()
            )));
        }
        self.ids.clear();
        self.suffix.clear();
        self.ids.push(cfg.bos);
        let mut ids = std::mem::take(&mut self.ids);
        self.seg(&inputs::question_segment(question), &mut ids)?;
        self.n_base = cands.len();
        self.spans_tmp.clear();
        if cfg.arm == Arm::ExternalExecutor {
            self.seg(inputs::FACTS, &mut ids)?;
            for (i, f) in cands.iter().enumerate() {
                let s = ids.len();
                self.seg(&inputs::candidate_line(i, f), &mut ids)?;
                self.spans_tmp.push((s, ids.len()));
            }
        }
        self.seg(inputs::STEPS, &mut ids)?;
        self.ids = ids;
        self.has_regs = false;
        self.hop_key = None;
        match cfg.arm {
            Arm::Rrt => {
                self.l0_rows = 0;
                self.setup_memory(cands)?;
            }
            Arm::ExternalExecutor => {
                self.committed = 0;
                self.run_ee_prefix()?;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------------------------------------ rrt memory
    fn setup_memory(&mut self, cands: &[CandidateFields]) -> Result<()> {
        let m = self.m;
        let cfg = &m.cfg;
        let (d, k, slots) = (cfg.hidden, cfg.k, cfg.codec.slots());
        let r = cfg.records();
        let w = cfg.memory_width;
        if w != d + codec::ROLE_WIDTH {
            return Err(Error::Format(format!("memory width {w} != d + 4")));
        }
        self.mem_values.resize(r * slots * w, 0.0);
        self.mem_mask.resize(r * slots, false);
        // base half
        codec::encode_memory(
            cfg.codec,
            cands,
            None,
            k,
            self.tok,
            &m.emb,
            &mut self.mem_values[..k * slots * w],
            &mut self.mem_mask[..k * slots],
        )?;
        if r > k {
            self.mem_values[k * slots * w..].fill(0.0);
            self.mem_mask[k * slots..].fill(false);
        }
        let c = m
            .cond
            .as_ref()
            .ok_or_else(|| Error::Format("rrt without conditioner".into()))?;
        let ntok = r * slots + 1;
        self.mem_tokens.resize(ntok * d, 0.0);
        // memory tokens = memory_projection(values masked to 0) for every slot (padded slots = bias), + NULL
        c.mem_proj.forward(
            &m.kernels,
            &self.mem_values,
            r * slots,
            &mut self.mem_tokens,
            d,
            crate::kernels::Epi::STORE,
            &mut self.b.gemm,
        );
        self.mem_tokens[(ntok - 1) * d..].copy_from_slice(&c.null_memory);
        self.refresh_memory_kv(0, r)?;
        // stack-v2: the hop half is empty until a decision reads a hop manifest (key 0 = no hop)
        self.hop_key = Some(0);
        Ok(())
    }

    /// recompute K/V of all memory tokens and the pointer keys of records [r0, r1) (+ NULL)
    fn refresh_memory_kv(&mut self, r0: usize, r1: usize) -> Result<()> {
        let m = self.m;
        let cfg = &m.cfg;
        let (d, slots) = (cfg.hidden, cfg.codec.slots());
        let r = cfg.records();
        let ntok = r * slots + 1;
        let c = m.cond.as_ref().unwrap();
        let mut valid = std::mem::take(&mut self.valid);
        valid.clear();
        valid.extend_from_slice(&self.mem_mask[..r * slots]);
        valid.push(true);
        if self.mem_kv.kv.len() != ntok * 2 * d || r0 == 0 {
            m.cross_kv(
                &c.mem,
                &self.mem_tokens,
                ntok,
                &valid,
                &mut self.mem_kv,
                &mut self.b,
            );
        } else {
            // only the changed records' tokens: K | V written in place (the projection is row-wise)
            let t0 = r0 * slots;
            let t1 = r1 * slots;
            c.mem.kv.forward(
                &m.kernels,
                &self.mem_tokens[t0 * d..t1 * d],
                t1 - t0,
                &mut self.mem_kv.kv[t0 * 2 * d..t1 * 2 * d],
                2 * d,
                crate::kernels::Epi::STORE,
                &mut self.b.gemm,
            );
            self.mem_kv.valid.clear();
            self.mem_kv.valid.extend((0..ntok).filter(|&i| valid[i]));
            self.mem_kv.compact(d);
        }
        self.valid = valid;
        // pointer keys: mean of the projected valid slots per record, NULL last
        self.ptr_keys.resize((r + 1) * d, 0.0);
        self.rec_valid.resize(r, false);
        for rec in r0..r1 {
            let key = &mut self.ptr_keys[rec * d..(rec + 1) * d];
            key.fill(0.0);
            let mut cnt = 0f32;
            for s in 0..slots {
                if self.mem_mask[rec * slots + s] {
                    let t = &self.mem_tokens[(rec * slots + s) * d..(rec * slots + s + 1) * d];
                    for i in 0..d {
                        key[i] += t[i];
                    }
                    cnt += 1.0;
                }
            }
            let cnt = cnt.max(1.0);
            for v in key.iter_mut() {
                *v /= cnt;
            }
            self.rec_valid[rec] = (0..slots).any(|s| self.mem_mask[rec * slots + s]);
        }
        self.ptr_keys[r * d..].copy_from_slice(&c.null_memory);
        Ok(())
    }

    /// stack-v2: encode the hop manifest into memory records K..2K (no-op when unchanged)
    fn set_hop_memory(&mut self, hop: Option<&HopInput>) -> Result<()> {
        let m = self.m;
        let cfg = &m.cfg;
        if !cfg.stack.has_hops() {
            return Ok(());
        }
        let key = hop.map(|h| h.key).unwrap_or(0);
        if self.hop_key == Some(key) {
            return Ok(());
        }
        let (d, k, slots) = (cfg.hidden, cfg.k, cfg.codec.slots());
        let w = cfg.memory_width;
        let (vals, mask) = (
            &mut self.mem_values[k * slots * w..2 * k * slots * w],
            &mut self.mem_mask[k * slots..2 * k * slots],
        );
        match hop {
            Some(h) => codec::encode_memory(
                cfg.codec,
                h.cands,
                Some(h.resolves),
                k,
                self.tok,
                &m.emb,
                vals,
                mask,
            )?,
            None => {
                vals.fill(0.0);
                mask.fill(false);
            }
        }
        let c = m.cond.as_ref().unwrap();
        c.mem_proj.forward(
            &m.kernels,
            &self.mem_values[k * slots * w..2 * k * slots * w],
            k * slots,
            &mut self.mem_tokens[k * slots * d..2 * k * slots * d],
            d,
            crate::kernels::Epi::STORE,
            &mut self.b.gemm,
        );
        self.refresh_memory_kv(k, 2 * k)?;
        self.hop_key = Some(key);
        Ok(())
    }

    /// register feedback of the executed stack (None at decision 0)
    fn set_registers(&mut self, stack: Option<&[StackEntry]>) {
        let m = self.m;
        let d = m.cfg.hidden;
        match stack {
            None => self.has_regs = false,
            Some(st) => {
                let mut f = [0f32; REGISTER_COUNT * REGISTER_WIDTH];
                let mut mask = [false; REGISTER_COUNT];
                codec::register_features(st, &mut f, &mut mask);
                self.set_register_features(&f, &mask);
                let _ = d;
            }
        }
    }

    /// register tokens from precomputed comparison16 features
    pub fn set_register_features(&mut self, f: &[f32], mask: &[bool]) {
        let m = self.m;
        let d = m.cfg.hidden;
        let c = m.cond.as_ref().unwrap();
        let mut raw = [0f32; REGISTER_COUNT * REGISTER_WIDTH];
        for s in 0..REGISTER_COUNT {
            if mask[s] {
                raw[s * REGISTER_WIDTH..(s + 1) * REGISTER_WIDTH]
                    .copy_from_slice(&f[s * REGISTER_WIDTH..(s + 1) * REGISTER_WIDTH]);
            }
        }
        self.reg_tokens.resize((REGISTER_COUNT + 1) * d, 0.0);
        c.reg_proj.forward(
            &m.kernels,
            &raw,
            REGISTER_COUNT,
            &mut self.reg_tokens,
            d,
            crate::kernels::Epi::STORE,
            &mut self.b.gemm,
        );
        self.reg_tokens[REGISTER_COUNT * d..].copy_from_slice(&c.null_register);
        let mut valid = [true; REGISTER_COUNT + 1];
        valid[..REGISTER_COUNT].copy_from_slice(&mask[..REGISTER_COUNT]);
        m.cross_kv(
            &c.reg,
            &self.reg_tokens,
            REGISTER_COUNT + 1,
            &valid,
            &mut self.reg_kv,
            &mut self.b,
        );
        self.has_regs = true;
    }

    /// token ids of several text segments, each tokenized separately and concatenated
    pub fn tokenize_segments(&mut self, segs: &[String]) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for s in segs {
            self.seg(s, &mut ids)?;
        }
        Ok(ids)
    }

    /// intra-question parallelism: run the sharded linears and the attention heads on this pool (None = one thread).
    /// The model must be loaded with `LoadOptions::shards` > 1 for the linears to split; results are bit-identical.
    pub fn set_pool(&mut self, pool: Option<std::sync::Arc<crate::pool::Pool>>) {
        self.b.gemm.pool = pool;
    }

    /// run the dual-format linears with their f32 weights (int8 guard fallback) until reset
    pub fn set_force_f32(&mut self, on: bool) {
        self.b.gemm.force_f32 = on;
    }

    pub fn force_f32(&self) -> bool {
        self.b.gemm.force_f32
    }

    /// start / stop recording per-channel input maxima of every f32 linear (int8 calibration)
    pub fn set_calibration(&mut self, on: bool) {
        self.b.gemm.calib = if on {
            Some(vec![Vec::new(); self.m.linear_names.len()])
        } else {
            None
        };
    }

    /// recorded per-channel maxima by linear name
    pub fn calibration(&self) -> Vec<(String, Vec<f32>)> {
        match &self.b.gemm.calib {
            Some(c) => self
                .m
                .linear_names
                .iter()
                .cloned()
                .zip(c.iter().cloned())
                .filter(|(_, v)| !v.is_empty())
                .collect(),
            None => Vec::new(),
        }
    }

    /// merge recorded maxima into `acc`
    pub fn merge_calibration(&self, acc: &mut std::collections::HashMap<String, Vec<f32>>) {
        for (n, v) in self.calibration() {
            let e = acc.entry(n).or_insert_with(|| vec![0.0; v.len()]);
            for (a, b) in e.iter_mut().zip(&v) {
                *a = a.max(*b);
            }
        }
    }

    /// memory values [R, slots, d+4] currently encoded (parity checks)
    pub fn memory_values(&self) -> &[f32] {
        &self.mem_values
    }

    /// register tokens are present for the next decision
    pub fn has_registers(&self) -> bool {
        self.has_regs
    }

    // ------------------------------------------------------------------------------------------------ decisions
    /// Run one decision. `history` = the history segment of the most recent action (None at decision 0); it is committed to
    /// the shared prefix. `stack`: the executed stack for the rrt registers (None at decision 0). `hop`: stack-v2 hop
    /// manifest. Returns op logits and pointer logits.
    pub fn decide(
        &mut self,
        history: Option<&str>,
        stack: Option<&[StackEntry]>,
        hop: Option<&HopInput>,
    ) -> Result<DecisionOut> {
        let mut new_ids = Vec::new();
        if let Some(h) = history {
            self.seg(h, &mut new_ids)?;
        }
        self.decide_ids(&new_ids, stack, hop, None)
    }

    /// as `decide`, with the history tokens given; `regs` (rrt) overrides the register features computed from `stack`
    pub fn decide_ids(
        &mut self,
        new_ids: &[u32],
        stack: Option<&[StackEntry]>,
        hop: Option<&HopInput>,
        regs: Option<(&[f32], &[bool])>,
    ) -> Result<DecisionOut> {
        let m = self.m;
        let cfg = &m.cfg;
        let arm = cfg.arm;
        let stackp = cfg.stack;
        // suffix tokens
        let mut suffix = std::mem::take(&mut self.suffix);
        suffix.clear();
        self.spans_tmp.clear();
        let hop_block = arm == Arm::ExternalExecutor
            && stackp.has_hops()
            && hop.is_some_and(|h| !h.cands.is_empty());
        if hop_block {
            let h = hop.unwrap();
            let k = cfg.k;
            self.seg(inputs::HOP_HEADER, &mut suffix)?;
            for (j, f) in h.cands.iter().enumerate() {
                let s = suffix.len();
                self.seg(
                    &inputs::hop_line(k + j, f, h.resolves.get(j).copied().unwrap_or(false)),
                    &mut suffix,
                )?;
                self.spans_tmp.push((s, suffix.len()));
            }
            self.seg(inputs::HOP_NEXT, &mut suffix)?;
        } else {
            self.seg(inputs::NEXT, &mut suffix)?;
        }
        self.suffix = suffix;
        let total = self.ids.len() + new_ids.len() + self.suffix.len();
        if total > cfg.max_positions {
            return Err(Error::Input(format!(
                "decision input of {total} tokens exceeds {}",
                cfg.max_positions
            )));
        }
        match arm {
            Arm::Rrt => {
                if let Some((f, m)) = regs {
                    self.set_register_features(f, m);
                } else {
                    self.set_registers(stack);
                }
                self.set_hop_memory(hop)?;
                self.decide_rrt(new_ids)
            }
            Arm::ExternalExecutor => self.decide_ee(new_ids),
        }
    }

    fn decide_rrt(&mut self, new_ids: &[u32]) -> Result<DecisionOut> {
        let m = self.m;
        let cfg = &m.cfg;
        let d = cfg.hidden;
        let nl = cfg.layers;
        self.ids.extend_from_slice(new_ids);
        let t = self.ids.len() + self.suffix.len();
        let r0 = self.l0_rows.min(self.ids.len());
        let n_new = t - r0;
        // layer 0 for the rows not cached yet (pre-hook output is exactly shared across decisions)
        grow(&mut self.out0, t * d);
        grow(&mut self.x, n_new * d);
        let mut all = Vec::with_capacity(n_new);
        all.extend_from_slice(&self.ids[r0..]);
        all.extend_from_slice(&self.suffix);
        m.embed(&all, &mut self.x[..n_new * d]);
        m.layer_forward(
            0,
            &mut self.x,
            n_new,
            r0,
            &mut self.kv[0],
            None,
            &mut self.xc,
            &mut self.b,
        );
        self.out0[r0 * d..t * d].copy_from_slice(&self.x[..n_new * d]);
        self.l0_rows = self.ids.len();
        // hook after layer 0, then layers 1..L-1 over all positions
        grow(&mut self.x, t * d);
        self.x[..t * d].copy_from_slice(&self.out0[..t * d]);
        let regs = if self.has_regs {
            Some(&self.reg_kv)
        } else {
            None
        };
        m.condition(&mut self.x[..t * d], t, &self.mem_kv, regs, &mut self.b);
        for l in 1..nl - 1 {
            m.layer_forward(
                l,
                &mut self.x,
                t,
                0,
                &mut self.kv[l],
                None,
                &mut self.xc,
                &mut self.b,
            );
            m.condition(&mut self.x[..t * d], t, &self.mem_kv, regs, &mut self.b);
        }
        // last layer: K/V of every position, everything else for the decision position only
        let last = [t - 1];
        m.layer_forward(
            nl - 1,
            &mut self.x,
            t,
            0,
            &mut self.kv[nl - 1],
            Some(&last),
            &mut self.xc,
            &mut self.b,
        );
        m.condition(&mut self.xc[..d], 1, &self.mem_kv, regs, &mut self.b);
        m.final_norm(&mut self.xc[..d], 1, &mut self.b);
        let mut out = DecisionOut {
            op_logits: vec![0.0; cfg.n_ops],
            ptr_logits: Vec::new(),
            computed_tokens: n_new + (nl - 1) * t,
            input_tokens: t,
        };
        grow(&mut self.q, d);
        grow(&mut self.lastv, d);
        self.lastv[..d].copy_from_slice(&self.xc[..d]);
        m.heads(
            &self.lastv[..d],
            &mut out.op_logits,
            &mut self.q[..d],
            &mut self.b,
        );
        let r = cfg.records();
        let div = cfg.pointer_divisor();
        out.ptr_logits.resize(r + 1, 0.0);
        for rec in 0..=r {
            out.ptr_logits[rec] = if rec == r || self.rec_valid[rec] {
                ops::dotf(&self.q[..d], &self.ptr_keys[rec * d..(rec + 1) * d]) / div
            } else {
                f32::MIN
            };
        }
        Ok(out)
    }

    // ------------------------------------------------------------------------------------------------ external executor
    fn run_ee_prefix(&mut self) -> Result<()> {
        let m = self.m;
        let cfg = &m.cfg;
        let d = cfg.hidden;
        let nl = cfg.layers;
        let p = self.ids.len();
        grow(&mut self.x, p * d);
        let ids = std::mem::take(&mut self.ids);
        m.embed(&ids, &mut self.x[..p * d]);
        self.ids = ids;
        for l in 0..nl - 1 {
            m.layer_forward(
                l,
                &mut self.x,
                p,
                0,
                &mut self.kv[l],
                None,
                &mut self.xc,
                &mut self.b,
            );
        }
        self.need.clear();
        for &(s, e) in &self.spans_tmp {
            self.need.extend(s..e);
        }
        let need = std::mem::take(&mut self.need);
        m.layer_forward(
            nl - 1,
            &mut self.x,
            p,
            0,
            &mut self.kv[nl - 1],
            Some(&need),
            &mut self.xc,
            &mut self.b,
        );
        let nn = need.len();
        m.capacity(&mut self.xc[..nn * d], nn, &mut self.b);
        m.final_norm(&mut self.xc[..nn * d], nn, &mut self.b);
        // base pointer keys = mean of each line's final hidden states
        self.base_keys.resize(self.spans_tmp.len() * d, 0.0);
        let mut row = 0;
        for (i, &(s, e)) in self.spans_tmp.iter().enumerate() {
            let key = &mut self.base_keys[i * d..(i + 1) * d];
            key.fill(0.0);
            for _ in s..e {
                let h = &self.xc[row * d..(row + 1) * d];
                for j in 0..d {
                    key[j] += h[j];
                }
                row += 1;
            }
            let len = (e - s) as f32;
            for v in key.iter_mut() {
                *v /= len;
            }
        }
        self.need = need;
        self.committed = p;
        Ok(())
    }

    fn decide_ee(&mut self, new_ids: &[u32]) -> Result<DecisionOut> {
        let m = self.m;
        let cfg = &m.cfg;
        let d = cfg.hidden;
        let nl = cfg.layers;
        let mut computed = 0;
        // 1. commit the new history segment (K/V of every layer; the last layer needs no outputs for it)
        let nh = new_ids.len();
        if nh > 0 {
            grow(&mut self.x, nh * d);
            m.embed(new_ids, &mut self.x[..nh * d]);
            let pos0 = self.committed;
            for l in 0..nl - 1 {
                m.layer_forward(
                    l,
                    &mut self.x,
                    nh,
                    pos0,
                    &mut self.kv[l],
                    None,
                    &mut self.xc,
                    &mut self.b,
                );
            }
            m.layer_forward(
                nl - 1,
                &mut self.x,
                nh,
                pos0,
                &mut self.kv[nl - 1],
                Some(&[] as &[usize]),
                &mut self.xc,
                &mut self.b,
            );
            self.ids.extend_from_slice(new_ids);
            self.committed += nh;
            computed += nh;
        }
        // 2. the suffix (not committed): hop lines + the decision position
        let ns = self.suffix.len();
        let pos0 = self.committed;
        grow(&mut self.x, ns * d);
        let suffix = std::mem::take(&mut self.suffix);
        m.embed(&suffix, &mut self.x[..ns * d]);
        self.suffix = suffix;
        for l in 0..nl - 1 {
            m.layer_forward(
                l,
                &mut self.x,
                ns,
                pos0,
                &mut self.kv[l],
                None,
                &mut self.xc,
                &mut self.b,
            );
        }
        self.need.clear();
        for &(s, e) in &self.spans_tmp {
            self.need.extend(s..e);
        }
        self.need.push(ns - 1);
        let need = std::mem::take(&mut self.need);
        m.layer_forward(
            nl - 1,
            &mut self.x,
            ns,
            pos0,
            &mut self.kv[nl - 1],
            Some(&need),
            &mut self.xc,
            &mut self.b,
        );
        let nn = need.len();
        m.capacity(&mut self.xc[..nn * d], nn, &mut self.b);
        m.final_norm(&mut self.xc[..nn * d], nn, &mut self.b);
        self.need = need;
        computed += ns;
        // hop keys
        let nhop = self.spans_tmp.len();
        self.hop_keys.resize(nhop * d, 0.0);
        let mut row = 0;
        for (i, &(s, e)) in self.spans_tmp.iter().enumerate() {
            let key = &mut self.hop_keys[i * d..(i + 1) * d];
            key.fill(0.0);
            for _ in s..e {
                let h = &self.xc[row * d..(row + 1) * d];
                for j in 0..d {
                    key[j] += h[j];
                }
                row += 1;
            }
            let len = (e - s) as f32;
            for v in key.iter_mut() {
                *v /= len;
            }
        }
        grow(&mut self.lastv, d);
        self.lastv[..d].copy_from_slice(&self.xc[(nn - 1) * d..nn * d]);
        let mut out = DecisionOut {
            op_logits: vec![0.0; cfg.n_ops],
            ptr_logits: Vec::new(),
            computed_tokens: computed,
            input_tokens: self.ids.len() + self.suffix.len(),
        };
        grow(&mut self.q, d);
        m.heads(
            &self.lastv[..d],
            &mut out.op_logits,
            &mut self.q[..d],
            &mut self.b,
        );
        // arms.Arm.decide: einsum(q, keys) / math.sqrt(d)
        let div = cfg.pointer_divisor();
        let nb = self.n_base;
        let null = m.null_key.as_ref().unwrap();
        let q = &self.q[..d];
        let dotk = |key: &[f32]| ops::dotf(q, key) / div;
        match cfg.stack {
            StackProtocol::V1 => {
                out.ptr_logits.resize(nb + 1, 0.0);
                for i in 0..nb {
                    out.ptr_logits[i] = dotk(&self.base_keys[i * d..(i + 1) * d]);
                }
                out.ptr_logits[nb] = dotk(null);
            }
            StackProtocol::V2 | StackProtocol::V5 => {
                let k = cfg.k;
                out.ptr_logits.clear();
                out.ptr_logits.resize(2 * k + 1, f32::MIN);
                for i in 0..nb {
                    out.ptr_logits[i] = dotk(&self.base_keys[i * d..(i + 1) * d]);
                }
                for j in 0..nhop.min(k) {
                    out.ptr_logits[k + j] = dotk(&self.hop_keys[j * d..(j + 1) * d]);
                }
                out.ptr_logits[2 * k] = dotk(null);
            }
        }
        Ok(out)
    }
}
