# xerj-infer — Rust CPU inference for the XERJ-R executor models

`xerj-infer` runs the trained XERJ-R decision models on CPUs without Python, PyTorch or transformers: the 17M `rrt` and
`external_executor` models (stack-v1 and stack-v2) and, since 2026-10-26, the 68M "direction" text executor
(`external_executor`, Ettin-68M, stack protocol v5 = stack-v2 + traversal direction; section
[Stack v5 and the 68M direction executor](#stack-v5-and-the-68m-direction-executor-2026-10-26)). It reproduces the
PyTorch forward pass bit-for-bit in its inputs and to f32 rounding noise in its outputs (exact f32 path), and adds a faster
int8 path: SIMD GEMM kernels selected at run time, an f32 recompute of close decisions, and prefix-cache reuse across the
decisions of one question.

Status 2026-10-26, stack v5 / 68M direction executor (`x68_dir_v1/external_executor_s11`, the run host, one X925 core):

* **f32 parity.** 480 golden questions, 1,433 decisions (direction test with 112 inverse hops, FactConsolidation MH 6K,
  MQuAKE-T chains): every decision input, every action and every free-running final record equals PyTorch's; the largest
  logit difference is 1.4e-5 (op) and 6.5e-5 (pointer, logits up to 42).
* **int8.** Raw, smoothed and guarded int8 all reproduce every action and final record; the guard recomputes 1.9–2.7 % of
  the decisions in f32 (one FactConsolidation decision exceeds the guard's error bound without flipping: see below).
* **Latency, p50 per question.** int8 + guard 237–259 ms against 1.6–3.1 s for PyTorch on the same core (6.7–11.8×); f32
  745–1,126 ms.
* **Memory.** 73 MB RSS after load (int8), 199 MB (f32), 243 MB (int8 + guard); PyTorch 1.4 GB.
* The 17M parity suite below is unchanged with this build, number for number.

Status 2026-10-21 (the run host, Cortex-X925):

* **f32 parity.** Over 2,000 golden questions (4,988 decisions; MQuAKE-T single-hop, MQuAKE-T multi-hop chains and the
  synthetic v1 test; 4 checkpoints):
  * the decision input ids and actions match PyTorch on every decision;
  * every free-running final answer matches;
  * the largest logit difference is 2.3e-5 (op) and 2.1e-4 (pointer). The pointer logits have magnitudes of about 110,
    so 2.1e-4 is about 2e-6 relative.
* **int8 + guard.** Every action and every final answer matches PyTorch's f32 decisions (4,988/4,988 and 2,000/2,000).
  Without the guard, 99.66 % of actions match: 100 % on MQuAKE, 97.3–99.9 % on the synthetic near-ties.
* **Latency, one thread, MQuAKE-T, p50.**
  * rrt: 2.0 ms per question, against 51 ms for PyTorch on the same questions and core (25×).
  * external_executor: 11–13 ms, against 104 ms (8×).
  * f32 path: 7.5 ms for rrt and 38 ms for external_executor.
* **Memory.** About 33–50 MB RSS after loading and 75–130 MB peak, against about 950 MB for PyTorch.

## Contents

| path | what |
|---|---|
| `src/config.rs` | export config (`config.json`): arm, stack protocol, codec, backbone shapes |
| `src/safetensors.rs` | minimal mmap safetensors reader (zero-copy f32 views, page release after load) |
| `src/model.rs` | weights in run-time layout, `Linear` (f32 / int8 / smoothing / dual weights for the guard), decoder layer, conditioner, capacity block, heads, KV cache, int8 calibration file |
| `src/session.rs` | one question: decision inputs, memory and register state, exact prefix reuse, decision forward |
| `src/episode.rs` | free-running driver (`rollout._execute` / `_execute_v2`), `ActionExecutor` trait, op / pointer selection, int8 guard |
| `src/inputs.rs`, `src/codec.rs`, `src/tokenizer.rs` | decision text (`inputs.py`, `stack_v2.py`), C1 / C1b memory codec + comparison16 registers (`xr_codecs.py`), HF tokenizer |
| `src/kernels/` | `mod.rs` (packing, dispatch), `neon.rs`, `x86.rs`, `scalar.rs`, `ops.rs` (LayerNorm, erf / GELU, exp, RoPE, attention), `kleidi.rs` (optional KleidiAI FFI) |
| `src/fixtures.rs` | golden fixtures from PyTorch + the replay executor |
| `src/bin/xerj-infer.rs` | CLI: `info`, `parity`, `calibrate`, `bench`, `gemmbench` |
| `tests/kernels.rs` | SIMD kernels vs scalar on random data (f32: 2e-5 relative; int8: bit-identical) |
| `tests/parity.rs` | golden parity vs PyTorch (runs when `XERJ_INFER_DATA` points at the exported data) |
| `third_party/kleidiai/` | vendored Arm KleidiAI subset (Apache-2.0) for the optional `kleidiai` feature |
| `research/xerj_r_mvp/tools/xerj_infer_export.py` | checkpoint → `model.safetensors` + `config.json` + `tokenizer.json` |
| `research/xerj_r_mvp/tools/xerj_infer_golden.py` | golden fixtures + Python latency baseline (pass A: `answer_item` timing; pass B: instrumented capture) |
| `research/xerj_r_mvp/tools/xerj_infer_export_v5.py`, `xerj_infer_golden_v5.py` | the same for the dev_v5 code (stack v5, any Ettin size): export with the production evaluation's loader and fingerprint check; fixtures with inverse flags |
| `scripts/xerj_infer_v5/` (repo root) | the run host drivers of the v5 work: fixtures, 68M parity / calibration / latency, the 17M regression, result tables |
| `remote_v5infer.sh` (repo root) | `remote.sh` for the v5-infer branch: own mirror and target dir, shared build lock, offline cargo |

## Architecture

**The model reproduced** (transformers 4.57.1 `modeling_modernbert_decoder.py`, `reference_pkg` `ettin_adapter.py` /
`model_reference.py`, `arms.py`, `trainer.load_trained`):

* **Embeddings.** `LayerNorm(E[id])` with no bias; layer 0 has no attention norm.
* **Attention.**
  * q / k / v / Wo carry no bias, and RoPE uses rotate-half with θ = 160000 and the exported `inv_freq`.
  * The scale is 1/8.
  * Layers 0, 3 and 6 are global causal. The other layers use a sliding window: query i sees keys i − 64 < j ≤ i.
* **MLP.** `Wo(gelu(x·Wiᵀ[:384]) · x·Wiᵀ[384:])` with exact erf GELU and pre-norm residuals.
* **rrt conditioner.** One shared `ReferenceConditioner` runs after every layer on every position:
  * `h += tanh(g_m)·MHA(LN_m(h), memory)`, then
  * `h += tanh(g_r)·MHA(LN_r(h), registers)` when registers exist.
  * Memory is C1b: the subject and relation phrase with a leading space, mean-pooled input embeddings at RMS 1, plus a role one-hot, giving 32 tokens. Stack-v2 adds 32 hop tokens with the resolve flag in role column 3, plus a NULL token.
  * The pointer is `pointer_query(last) · pooled_record_key / 16`, with masked records at `finfo.min`.
* **external_executor.**
  * A capacity block runs after the last layer: `h + up(gelu(down(LN(h))))` with a bottleneck of 1166.
  * Pointer keys are the means of each candidate line's final hidden states, plus a learned NULL key.
  * Stack-v2 remaps the keys to the `[base K | hop K | NULL]` layout.
* **Not run.** The LM head is removed for executor arms. PyTorch's `forward_step` still computes the unused `hidden @ Eᵀ`
  logits over all positions; xerj-infer does not.

**Exact prefix reuse** (`session.rs`). The decision inputs of one question are `P + H_0..H_{t-1} + suffix_t`:

* P is the question, plus the candidate lines and `Steps:` for the external executor.
* Each executed action appends one history segment H.
* The suffix is `\nNext:`, or the stack-v2 hop block.

What is shared, and therefore cached:

* **external_executor.** The model is a plain causal decoder plus a position-wise block, so the K/V of P + H at every
  layer are exact for all later decisions.
  * P runs once per question; the base pointer keys come from it.
  * Each decision appends its new H to the cache, as K/V only for the last layer, and runs just its suffix.
  * Last-layer attention, MLP and capacity block run only for the rows that are read: the line spans and the decision
    token.
* **rrt.** Register attention changes every position after every layer and every action, so layers 1–6 must be
  recomputed for all positions. Verified in `ettin_adapter._condition`: the hook runs on the whole sequence. What is
  shared and cached:
  * embeddings, layer-0 K/V and the layer-0 pre-hook outputs of P + H;
  * the memory tokens' K/V, which are the same at every layer because the conditioner is shared. They are computed once
    per question, and only the hop half is recomputed when the stack-v2 hop manifest changes.
  * The last layer runs everything after its K/V for the decision position only.

**Executor boundary.** The exact stack runner, store and manifests are not part of this crate. `episode::ActionExecutor`
provides:

* the legal ops;
* `step(op, pointer)`;
* the trace (op, pointer and result text, for the external executor's history);
* the typed stack, which the engine turns into comparison16 registers itself;
* the stack-v2 hop manifest.

Inputs are the question and the four `candidate_fields` strings per candidate, which the store layer renders. Stack v5 adds
two flags and nothing else: `CandidateFields::inverse` on the inverse candidates of the base and hop manifests and
`HistoryStep::inverse` on executed inverse hops (trace `"direction": "in"`). The parity
tests use `fixtures::ReplayExecutor`, which answers from the recorded PyTorch episode.

## Kernels

All GEMMs compute `y = epilogue(x·Wᵀ + b)`. The epilogue fuses the bias, a scale α and an optional accumulate into `y`,
which covers residual adds and the conditioner's gated residual `h += tanh(g)·(…)`. The following are fused or not
materialized:

* QKV is one GEMM, N = 768.
* GeGLU input and gate are one GEMM.
* Under int8, LayerNorm is fused into activation quantization.
* RoPE runs in place and writes the KV cache directly.

| path | weights | kernels (runtime dispatch) |
|---|---|---|
| f32 | panels of 16 output channels, `[N/16][K][16]` | NEON 6×16 (FMLA by element, A packed `[k][6]`); AVX2+FMA 6×16; AVX-512F 12×16; scalar |
| int8 | per-output-channel symmetric int8, one layout for every ISA: blocks of 8 channels × 8 k (64 B) | NEON **i8mm** 8×8 tiles (SMMLA, 16 accumulators); NEON **dotprod** 4×8 (SDOT on row-duplicated A, pairwise lane sums); AVX-512 **VNNI** 8×8 (VPDPBUSD on a+128, corrected with per-channel Σw); **AVX2** 4×8 (VPMADDUBSW with the sign trick — exact for ±127 operands); scalar |
| attention f32 | K cache also transposed `[d][T+16]` | NEON 4 queries × 16 keys by-element FMLA (no horizontal reductions), vectorized softmax (polynomial exp), P transposed so P·V is one FMLA per key and lane |
| attention, int8 path | bf16 copies of K (pair layout) and V (transposed pair layout) | NEON **BFMMLA** (FEAT_BF16): Q·Kᵀ 4 × 16 keys and P·V, f32 accumulation; f32 path keeps f32 attention |

Activations are quantized per row (symmetric, round-half-away). An optional per-channel pre-scale implements
SmoothQuant-style smoothing: `s_j = max|X_j|^α / max|W_j|^(1−α)` from dev calibration, with the weights multiplied by
`s` and the activations by `1/s`.

Element-wise kernels:

* LayerNorm (two-pass).
* GELU, through a rational erf approximation (|err| < 2e-7, bit-identical scalar / NEON).
* RoPE, rounded like PyTorch: separate products, then the sum.

`cargo test` checks:

* the f32 SIMD kernels against scalar, within 2e-5 relative;
* every int8 kernel against scalar, bit-identical, including smoothing and fused-LayerNorm quantization.

**x86.** AVX2 + FMA (f32) and AVX2 (int8) were tested under qemu-user 7.2 `-cpu max`, with a cross-built static musl
binary run in a Debian container: all kernel tests pass. The AVX-512F and AVX-512 VNNI kernels compile but are
**untested**: qemu's TCG has no AVX-512, and no x86 hardware was available.

**SVE2** is present on X925 but has 128-bit vectors. NEON i8mm / bf16 cover the same operations, so there is no SVE path.

### int8 accuracy: smoothing + guard

Plain W8A8 (per-channel weights, per-row activations) already gives the same decisions as PyTorch on all MQuAKE
questions. It fails only on near-ties in the synthetic v1 set: f32 pointer margins of 0.007–1.0, between two relations of
the same subject. Two mechanisms close the gap:

1. **Calibration** (`xerj-infer calibrate`, on DEV-split fixtures, never on the test fixtures):
   * chooses α ∈ {0.3, 0.5, 0.7} by the smallest pointer-logit error;
   * measures the largest int8-vs-f32 logit error ε of the op and pointer heads on the calibration decisions;
   * writes `int8_calib.json`.
2. **Guard** (`episode::run_episode_guarded`).
   * **Trigger:** an int8 decision whose winning margin is below τ = 2 × 2.0 × ε is recomputed with the f32 weights,
     from scratch. The recompute uses the prefix and the full history, so its result is exactly the f32 decision. If all
     errors stay below 2ε, any decision with margin > 4ε cannot flip.
   * **Afterwards:** the question continues in int8 on top of the f32-built caches.
   * **Headroom:** errors on the test sets reached at most 1.6 ε_dev.
   * **Cost:** the guard keeps the f32 weights next to the int8 ones, about +18 MB resident.

Calibrated thresholds (dev splits, 300 questions per model):

| model | α | ε op | ε pointer | τ op | τ pointer |
|---|---:|---:|---:|---:|---:|
| v1 rrt | 0.3 | 0.45 | 0.55 | 1.81 | 2.21 |
| v1 external_executor | 0.3 | 0.17 | 6.82 | 0.70 | 27.3 |
| v2 rrt | 0.5 | 0.61 | 0.54 | 2.44 | 2.14 |
| v2 external_executor | 0.3 | 0.41 | 3.40 | 1.65 | 13.6 |

### Borrowed from production engines: KleidiAI

The coordinator asked to compare against KleidiAI and ggml. Pinned copies are on the run host at `~/xerj_baselines/kleidiai`
@ 6233410. KleidiAI's W8A8 kernel `matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm` is per-row asymmetric
activations × per-channel weights, a hand-scheduled 16×4 SMMLA kernel. It was vendored (`third_party/kleidiai`,
feature `kleidiai`, `--kleidi on`) and timed in the same harness (`gemmbench`), one X925 core, µs per GEMM including
activation quantization:

| M×N×K | own SMMLA 8×8 | KleidiAI 16×4 |
|---|---:|---:|
| 430×768×256 | 231 | 215 |
| 430×256×384 | 136 | 124 |
| 430×1166×256 | 359 | 326 |
| 35×768×256 | 20.6 | 19.3 |
| 8×768×256 | **4.7** | 6.7 |
| 1×768×256 | **2.3** | 3.1 |

In isolation KleidiAI is 6–9 % faster at M ≥ 16 and slower at small M. End to end it is slower: rrt 2.16 vs 1.83 ms,
external_executor 14.5 vs 14.0 ms on MQuAKE-T (int8 without guard). Its packer needs the LayerNorm output and the
smoothed activations materialized, and residual and gated epilogues need an extra pass. Its asymmetric quantization did
not lower the logit error (ε pointer 0.50 vs 0.55 for v1 rrt).

**Decision:** the crate's own kernels stay the default; KleidiAI remains an opt-in backend. Ideas worth porting later:
16-row tiles with 4-row interleaved LHS packing and a software-pipelined asm K loop. ggml's Q8_0×Q8_0 i8mm/dotprod kernels
use per-32-block scales on both operands. That is more accurate but costs float work per block. It was not adopted,
because the guard already gives exact decisions.

## How to run

All builds and runs happen on the run host; the local disk is full. Large files live in `~/xerj_infer_data/`.

```bash
# 1. export a checkpoint (CPU; tools/ in research/xerj_r_mvp, run on the run host)
.venv-r/bin/python -B tools/xerj_infer_export.py --arm rrt --ckpt runs/mquake_ft/rrt_s11 --out ~/xerj_infer_data/weights/v1_rrt
#    -> model.safetensors (f32, tied LM decoder dropped), config.json, tokenizer.json

# 2. golden fixtures + Python baseline (one thread, one X925 core); --split dev for calibration data
XR_THREADS=1 taskset -c 7 .venv-r/bin/python -B tools/xerj_infer_golden.py --arm rrt --ckpt runs/mquake_ft/rrt_s11 \
    --set mquake_t --n 200 --out ~/xerj_infer_data/fixtures/v1_rrt.mquake_t.jsonl      # sets: mquake_t | mquake_t_chain | v1

# 3. build / test (from xerj_mem/)
./remote.sh cargo build --release -p xerj-infer                         # add --features kleidiai for the KleidiAI backend
./remote.sh cargo test --release -p xerj-infer                          # kernels vs scalar, codec / registers, erf / exp
./remote.sh XERJ_INFER_DATA=~/xerj_infer_data cargo test --release -p xerj-infer --test parity -- --nocapture

# 4. int8 calibration on DEV fixtures (writes <model>/int8_calib.json)
B=~/xerj_mem_target/release/xerj-infer; W=~/xerj_infer_data/weights; F=~/xerj_infer_data/fixtures; C=~/xerj_infer_data/calib
$B calibrate --model $W/v1_rrt --fixtures $C/v1_rrt.mquake_t.jsonl,$C/v1_rrt.v1.jsonl

# 5. parity and benchmarks
$B parity --model $W/v1_ee --fixtures $F/v1_ee.mquake_t.jsonl --precision f32
$B parity --model $W/v1_ee --fixtures $F/v1_ee.mquake_t.jsonl --precision int8 --int8-mode guard   # raw | smooth | guard
taskset -c 7 $B bench --model $W/v1_rrt --fixtures $F/v1_rrt.mquake_t.jsonl --precision int8 --rounds 3 [--threads 2]
XERJ_PROFILE=1 taskset -c 7 $B bench ...                                # per-component time breakdown
taskset -c 7 $B gemmbench                                              # kernel microbenchmark (+ KleidiAI when built in)

# 6. x86-64 kernels (cross-compiled, run under qemu-user in a container)
RUSTFLAGS="-C target-cpu=x86-64" CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=rust-lld \
  cargo test --release -p xerj-infer --no-default-features --target x86_64-unknown-linux-musl --no-run
docker run --rm -v <deps dir>:/t:ro debian:bookworm-slim sh -c 'apt-get update && apt-get install -y qemu-user && qemu-x86_64 -cpu max /t/kernels-<hash>'
```

**Environment switches:**

* `XERJ_PROFILE=1` turns on the profiler.
* `XERJ_BF16_ATTN=0` uses f32 attention in the int8 path.
* `XERJ_INT8_KEEP_F32=qkv,wo,…,layerN` keeps chosen int8 components in f32, for diagnostics.

**Rust API:**

```rust
let engine = xerj_infer::Engine::load_with(dir, &opts)?;
let mut s = engine.session();
xerj_infer::episode::run_episode_guarded(&mut s, question, &cands, &mut executor, None, guard)?;
```

## Stack v5 and the 68M direction executor (2026-10-26)

**Model.** `research/xerj_r_mvp/runs/x68_dir_v1/external_executor_s11` (production candidate of the XERJ-R readiness checklist):
`external_executor` on Ettin-68M (ModernBertDecoder, 19 layers, d = 512, 8 heads × 64, GeGLU 768, vocab 50,368, the shared
Ettin-17M tokenizer), capacity block bottleneck 2,318 after the last layer, K = 16, init identity, stack protocol **v5**
(dev_v5 `stack_v5.py`: stack-v2 + traversal direction), 4,000 steps; fingerprint `67f4454c…29620`, `model.pt` sha256
`3c1c95b4…49ea0` (both checked at export).

**What v5 means for the engine.** Direction is a property of the candidate, not an operation: the 10-way op head, the
legality masks, the pointer layout `[base 0..K-1 | hop K..2K-1 | NULL 2K]` and the hop blocks are stack-v2's. Three
model-visible strings change (all three are in the golden inputs and reproduced token for token):

* an inverse candidate line in the base block: `"[k] <= subject | relation | kind: value | quote\n"`;
* an inverse candidate in the hop block: `"[K+j] <= …"` (the `"<= "` marker replaces the resolve marker `"= "`; inverse hop
  candidates always resolve);
* an executed inverse hop in the history: `" LOOKUP [3] IN -> VALUE text Liberia;"`, `" CHAIN [18] IN -> …;"`.

Code: `config.rs` (`StackProtocol::V5`, `name()`, `has_hops()`; `rrt` + v5 is refused at load because no rrt v5
checkpoint exists to verify its memory codec against), `inputs.rs` (markers; unit test `v5_markers`), `episode.rs` /
`session.rs` (the stack-v2 paths for V5), `fixtures.rs` (inverse flags of the v5 fixtures). The pointer logits are now
divided by float32(√d) as `arms.Arm.decide` does (`einsum / math.sqrt(d)`), instead of multiplied by 1/√d: at d = 256 both
are exact, so the 17M numbers do not move; at d = 512 the division is the PyTorch rounding. Nothing else was 68M-specific:
every shape (19 layers, d 512, 8 heads, head_dim 64, intermediate 768, capacity 2,318, layer types, sliding window 64, RoPE
inv_freq) comes from the export; head_dim 64 keeps the NEON attention / RoPE / bf16 kernels; the capacity GEMMs (N or K =
2,318) use the padded panels as the 17M's 1,166.

**Executor interface (for the v5 executor port, branch v5-exec).** Two public fields, both `false` under stack v1 / v2:

```rust
pub struct CandidateFields { pub subject: String, pub relation: String, pub value: String, pub quote: String,
                             pub inverse: bool }   // stack_v5.InverseCand (base and hop manifests)
pub struct HistoryStep { pub op: String, pub pointer: Option<usize>, pub result: Option<String>,
                         pub inverse: bool }       // trace entry with "direction": "in"
```

Both structs now derive `Default`; `CandidateFields::new(..)` keeps `inverse = false`, `.with_inverse(bool)` sets it.
`ActionExecutor` and `HopView` keep their shape: a v5 adapter sets `inverse` on the base candidates it passes to
`run_episode` and on its hop candidates (`HopView::cands`; `resolves` = stack_v5's, true for forward and inverse tier-1
records), and on the `HistoryStep` of every action executed on an inverse candidate. `xerj-serve` was only adapted to
compile (`inverse: false` in `adapter.rs`; `engine.rs` refuses a v5 export until xerj-exec has the v5 executor).

**Export and golden fixtures (the run host, PyTorch CPU FP32, dev_v5 code imported read-only).**

* `tools/xerj_infer_export_v5.py`: loads the checkpoint exactly as the production evaluation (`tools/eval_generalist_v5.sh` →
  `trainer.load_trained` with the recorded `target_total` and backbone size: strict state_dict, fingerprint check), re-checks
  the fingerprint and the `model.pt` sha256, writes `model.safetensors` (283 MB f32, tied LM decoder dropped after a bit-equality
  check), `config.json` (format unchanged, `xerj-infer-export-v1`, plus optional `exporter` / `pointer` / `direction` keys,
  so 17M exports and old engines stay compatible), `tokenizer.json` and `tokenizer.json.sha256`. Output:
  `research/xerj_r_mvp/runs/xerj_infer_export_v5/x68_dir_s11/`.
* `tools/xerj_infer_golden_v5.py`: pass A = `rollout.answer_item(..., stack="v5", entity_ids=…)` as `evaluate_world` calls it
  (one torch thread pinned to X925 core 7: the single-thread baseline); pass B = the `_execute_v2` loop for v5 with the same
  stack_v5 functions, recording ids, spans, hop manifests with resolve and inverse flags, legality, op logits, layout and raw
  pointer logits, actions and trace entries with `direction`. Every pass-B final record (status, answer, evidence ids,
  program incl. `direction`, error) equals pass A's: 0 mismatches on all 5 sets. Fixture keys added: `cands_inverse`, hop
  `inverse`, trace `direction`.

**Golden sets** (`x68_dir_s11.<set>.jsonl`; the K = 16 v5 manifests and episodes exactly as `rollout.answer_item`):

| set | questions | decisions | with hop block | with inverse lines | inverse hops executed (LOOKUP IN / CHAIN IN) | CHAIN | max tokens | PyTorch statuses |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| direction test (`direction_v1` test, all items) | 134 | 324 | 168 | 297 | 112 (84 / 28) | 56 | 769 | ANSWER 112, UNKNOWN 22 |
| FactConsolidation MH 6K (all kept items) | 96 | 331 | 234 | 236 | 0 | 139 | 783 | ANSWER 95, UNKNOWN 1 |
| MQuAKE-T chains (`chain_stack_v2` test_t, 250) | 250 | 778 | 523 | 523 | 0 | 278 | 828 | ANSWER 245, UNKNOWN 5 |
| calibration (dev): direction dev + MQuAKE chain dev | 266 + 200 | 642 + 632 | 332 + 430 | | 222 + 0 | | | |

Every hop block of the v5 model carries inverse candidates (`<= ` lines), so all three sets test the markers; only the
direction set makes the model execute inverse hops. The direction set has 134 test items in total (one hub world), and
FactConsolidation MH 6K has 96 kept items, so both are used whole.

**f32 parity** (`xerj-infer parity --precision f32`, X925 core 7; forced = every recorded decision with the recorded history,
free-running = the engine's own actions drive the replay executor):

| set | decisions | input ids identical | max abs op logit diff | max abs pointer logit diff (largest abs pointer logit) | actions identical | inverse actions identical | free-running episodes (final answer records) identical |
|---|---:|---:|---:|---:|---:|---:|---:|
| direction test | 324 | 324/324 | 1.24e-5 | 3.43e-5 (41.6) | 324/324 | 112/112 | 134/134 |
| FactConsolidation MH 6K | 331 | 331/331 | 1.43e-5 | 3.58e-5 (39.9) | 331/331 | - | 96/96 |
| MQuAKE-T chains | 778 | 778/778 | 1.43e-5 | 6.48e-5 (39.7) | 778/778 | - | 250/250 |

`cargo test --test parity` with `XERJ_INFER_DATA` on these fixtures (first 25 questions per set, tolerance 5e-4, int8 + guard
identical episodes) passes.

**int8** (W8A8: per-channel int8 weights, per-row int8 activations, SMMLA; bf16 attention). Calibration (`xerj-infer
calibrate`) on the two DEV fixture sets (466 questions, 1,274 decisions): α = 0.7 (of 0.3 / 0.5 / 0.7: dev pointer error
5.06 / 3.98 / 2.66), ε_op = 0.63, ε_ptr = 2.66, safety 2 → τ_op = 2.52, τ_ptr = 10.66 (`int8_calib.json` in the export).

| set | mode | max abs op / pointer logit diff (forced) | actions identical | free-running episodes identical | f32 recomputes (decisions / questions) |
|---|---|---:|---:|---:|---:|
| direction test | raw | 0.442 / 2.291 | 324/324 | 134/134 | - |
| direction test | smooth | 0.253 / 1.548 | 324/324 | 134/134 | - |
| direction test | guard (shipped, safety 2) | 0.253 / 0.803 | 324/324 | 134/134 | 6 / 6 |
| direction test | guard, safety 3 | 0.242 / 0.803 | 324/324 | 134/134 | 77 / 77 |
| FactConsolidation MH 6K | raw | 0.374 / 3.793 | 331/331 | 96/96 | - |
| FactConsolidation MH 6K | smooth | 0.322 / 5.786 | 331/331 | 96/96 | - |
| FactConsolidation MH 6K | guard (shipped, safety 2) | 0.322 / 5.786 | 331/331 | 96/96 | 8 / 8 |
| FactConsolidation MH 6K | guard, safety 3 | 0.322 / 5.786 | 331/331 | 96/96 | 16 / 16 |
| MQuAKE-T chains | raw | 0.892 / 2.965 | 778/778 | 250/250 | - |
| MQuAKE-T chains | smooth | 0.366 / 1.273 | 778/778 | 250/250 | - |
| MQuAKE-T chains | guard (shipped, safety 2) | 0.339 / 1.273 | 778/778 | 250/250 | 21 / 20 |
| MQuAKE-T chains | guard, safety 3 | 0.339 / 1.273 | 778/778 | 250/250 | 42 / 37 |

Every int8 variant, even raw W8A8, reproduces every PyTorch action and every final record on the three sets; the shipped
guard recomputes 1.9 % / 2.4 % / 2.7 % of the decisions in f32. **Caveat:** on FactConsolidation one decision
(`factcons-6k/mh010` d0) has a smoothed-int8 pointer error of 5.79 (the next largest is 1.46), 2.2 ε_dev and above
τ_ptr / 2 = 5.33, so the guard's no-flip argument (errors below 2 ε) does not formally cover that set (it has no dev split to
calibrate on; smoothing α = 0.7 even raises its worst error from 3.79 raw). No decision flipped. Safety 3 (τ = 6 ε) covers all
three sets but recomputes 24 % of the direction decisions.

**Latency, one thread on X925 core 7, prefix reuse on, per question** (`xerj-infer bench`, warm-up pass + 1 measured round;
PyTorch = pass A of the golden tool: one torch thread on the same core, `answer_item` and the time inside `arm.decide`):

| set | xerj-infer f32 p50 / p95 / mean ms | int8 + guard p50 / p95 / mean ms (recomputes) | int8 smooth, no guard p50 / mean ms | PyTorch answer_item p50 / mean ms | PyTorch decide p50 / mean ms | decisions per question |
|---|---:|---:|---:|---:|---:|---:|
| direction test | 745 / 1,105 / 814 | **237** / 359 / 262 (6) | 229 / 247 | 1,597 / 1,924 | 1,586 / 1,908 | 2.42 |
| FactConsolidation MH 6K | 1,126 / 1,688 / 1,189 | **254** / 509 / 294 (8) | 248 / 265 | 2,905 / 3,255 | 2,878 / 3,225 | 3.45 |
| MQuAKE-T chains | 947 / 1,224 / 962 | **259** / 519 / 288 (21) | 256 / 263 | 3,065 / 3,151 | 3,055 / 3,141 | 3.11 |

int8 + guard is 6.7× / 11.3× / 11.8× faster than PyTorch's decide time at p50 (f32: 2.1× / 2.6× / 3.2×; the more decisions a
question takes, the more the prefix reuse saves). The first (cold) pass, with every segment tokenized for the first time, is
within a few percent of the warm round except int8 + guard on FactConsolidation (p50 338 vs 254 ms).

**Memory** (bench process, `VmRSS` after loading model + tokenizer / `VmHWM` peak over the run):

| precision | RSS after load | peak RSS | resident weights | embedding table (mmap, touched rows only) | load (model + tokenizer) |
|---|---:|---:|---:|---:|---:|
| f32 | 199 MB | 358–389 MB | 181 MB | 103 MB | 135 + 63 ms |
| int8 + guard (int8 + f32 weights) | 243 MB | 441–468 MB | 227 MB | 103 MB | 290 + 53 ms |
| int8 smooth, no guard (int8 weights only) | 73 MB | 268–297 MB | 48 MB | 103 MB | 125 + 44 ms |

PyTorch's golden process peaks at 1.43 GB. The peaks above the load RSS are mostly the per-layer KV caches (19 layers × K, Kᵀ,
V in f32, capacity rounded up to 1,024 positions ≈ 120 MB, plus the bf16 copies on the int8 path).

**17M regression.** With this branch's binary, the 2026-10-21 parity suite (`~/xerj_infer_data`, read only; 10 fixture sets,
4 checkpoints, 2,000 questions, 4,988 decisions, f32 and int8 + guard) gives results identical to the recorded ones in every
field: input ids, actions and free-running episodes all identical, the same largest logit differences to the last digit (e.g.
v2_ee MQuAKE-T chains f32 2.00e-5 / 5.48e-5) and the same guard recompute counts (`results_17m/`). `cargo test --test
parity` on the 17M data also passes.

**Commands** (the run host; the drivers are in `scripts/xerj_infer_v5/`, data in `~/xerj_mem_v5_models/`):

```bash
P=~/xerj-extractor/research/xerj_r_mvp/.venv-r/bin/python; MVP=~/xerj-extractor/research/xerj_r_mvp
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=
flock ~/xerj_mem_build.lock nice -n 15 $P -B tools/xerj_infer_export_v5.py --mvp $MVP --arm external_executor \
    --ckpt ckpt/x68_dir_v1/external_executor_s11 --out export/x68_dir_s11
scripts/make_fixtures_v5.sh          # 3 test sets (pass A 1 thread, core 7) + 2 dev sets, xerj_infer_golden_v5.py
./remote_v5infer.sh cargo build --release -p xerj-infer       # from the worktree: own mirror / target dir, lock, offline
./remote_v5infer.sh cargo test --release -p xerj-infer
./remote_v5infer.sh env XERJ_INFER_DATA=\$HOME/xerj_mem_v5_models/data cargo test --release -p xerj-infer --test parity -- --nocapture
scripts/run_v5_parity.sh f32|calib|int8|bench|guard3           # xerj-infer parity / calibrate / bench, one core (guard3: safety 3)
scripts/run_17m_parity.sh                                      # the unchanged 17M parity with this build
python3 scripts/v5_tables.py                                   # the tables above
```

## Results

* 17M: the status summary at the top; the full per-set tables of 2026-10-21 are in `~/xerj_infer_data/results`
  (`parity_table.py`, `bench_table.py` there).
* Stack v5 / 68M: the section above (raw results in `research/xerj_r_mvp/runs/xerj_infer_export_v5/`).

## Open issues

* **AVX-512 kernels are untested.** The AVX-512F (f32) and AVX-512 VNNI (int8) kernels are compile-checked only. qemu has
  no AVX-512 and there was no x86 hardware. AVX2 is emulator-tested, not benchmarked on hardware.
* **All-core benchmarks are pending.** Throughput was measured with 1 and 2 threads, the thermal cap on the run host while its
  GPU trains, and scales about 1.9–2.0× from 1 to 2 threads. 8-thread and all-core runs are pending.
* **The guard is expensive on near-tie data.**
  * On the synthetic v1 set 50–100 % of the external_executor questions are recomputed in f32, so int8 + guard costs
    f32 time there: v1 external_executor int8 + guard runs at the f32 latency.
  * On MQuAKE the recompute rate is 2–15 % of questions.
  * A more accurate int8 would lower τ and the recompute rate: per-group activation scales, or activations as two int8
    terms. So would a cheaper partial recompute.
* **The f32 path is compute-bound.** It runs at about 140 GFLOP/s, about 90 % of the measured FMLA peak. Further f32 gains
  need less work, not better kernels.
* **The int8 GEMM is about 15 % off the measured SMMLA peak.** It runs at about 800 GOPS, against 1350 peak. KleidiAI's
  schedule reaches 870 GOPS in isolation.
* **No intra-question multithreading yet.** The external_executor prefix is about 430 tokens and could be split across
  2–4 cores for latency.
* **File size.** The weight file is the 70 MB f32 export, of which 51.6 MB is the embedding table. It is memory-mapped,
  so only the touched rows become resident. A compact int8 export would be about 19 MB with an int8 embedding table or
  31 MB with f16. It is not built, because the exact f32 recompute needs the f32 table.
* **Tokenizer.** The reference HF tokenizer accounts for most of the 70–110 ms startup (`tokenizer.json` parse) and about
  10 MB RSS.
* **Executor integration.** The `ActionExecutor` implementation over the real store and executor (`xerj-exec`) is up to
  that crate. Here it is the fixture replay.
* **Stack v5 (2026-10-26).**
  * **No v5 executor yet.** xerj-exec (this branch) runs stack v1 / v2 only, so `xerj-serve` refuses v5 exports; the v5
    executor (inverse manifests, `EpisodeV5`) is ported on branch v5-exec. Its adapter must set `CandidateFields::inverse`
    on base and hop candidates and `HistoryStep::inverse` on executed inverse hops; nothing else changes at the interface.
  * **rrt + v5 is refused** at load: the memory codec would carry the inverse flag in role column 2 of the subject slot
    (`xr_codecs.c1b_memory`), but no rrt v5 checkpoint exists to check it against.
  * **Guard bound on FactConsolidation.** The int8 calibration has dev splits only for the direction and MQuAKE chain sets.
    On FactConsolidation MH 6K one decision has a pointer error of 5.79 = 2.2 ε_dev, above τ / 2, so the no-flip argument does
    not formally hold there (no decision flipped). Options: calibrate on more diverse dev data (a FactConsolidation-style dev
    world), a per-set safety (safety 3 costs 24 % recomputes on the direction set), or smoothing chosen by the worst set.
  * **Inverse hops are exercised by one set.** 112 of the 1,433 test decisions (and 222 of the 1,274 dev decisions) execute an
    inverse hop, all from the direction worlds (one hub world for test); the chain sets only carry inverse lines in their hop
    blocks.
  * **KV-cache memory at 68M.** 19 layers × (K, Kᵀ, V) f32 with power-of-two capacity is about 120 MB at 1,024 positions (plus
    about 40 MB of bf16 copies on the int8 path): the peak RSS is 268–468 MB against 73–243 MB after load. Exact-size or
    bf16 / f16 KV storage would roughly halve it.
  * **The f32 path at 68M** takes 0.75–1.1 s per question on one core; int8 is the serving path. Sizes other than 17M and
    68M (32M, 150M, 400M exports) were not run.
