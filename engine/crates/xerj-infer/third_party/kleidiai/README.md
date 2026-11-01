# Vendored Arm KleidiAI micro-kernels (subset)

Source: https://github.com/ARM-software/kleidiai @ 6233410e039011aa7422d0cd7c36d0f8c59af239 (2026-10-19), Apache-2.0
(`LICENSE-Apache-2.0.txt`, SPDX headers kept in every file). Files are unmodified copies.

Used by the optional `kleidiai` feature of xerj-infer (int8 GEMMs with M >= 16 on FEAT_I8MM CPUs):
* `kai_lhs_quant_pack_qai8dxp_f32` — per-row dynamic asymmetric int8 quantization + packing of the f32 activations
* `kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon` — packing of per-channel symmetric int8 weights (+ scales, bias)
* `kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm` — SMMLA 16x4 micro-kernel (hand-scheduled asm), f32 output
