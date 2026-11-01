//! FFI to the vendored Arm KleidiAI micro-kernels (feature `kleidiai`): per-row dynamic asymmetric int8 activations
//! (qai8dx) × per-channel symmetric int8 weights (qsi8cx), f32 output, SMMLA 16×4 (hand-scheduled asm). Used for the int8
//! GEMMs with M >= KAI_MIN_M on FEAT_I8MM CPUs, where it measured faster than the crate's own SMMLA kernel; smaller M keeps
//! the crate's kernel (2-row granularity).

use super::{AVec, Epi};
use std::ffi::c_void;

pub const KAI_MIN_M: usize = 16;
const MR: usize = 4;
const NR: usize = 4;
const KR: usize = 8;
const SR: usize = 1;

#[repr(C)]
struct RhsPackParams {
    lhs_zero_point: i32,
    scale_multiplier: f32,
}

extern "C" {
    fn kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        n: usize,
        k: usize,
        nr: usize,
        kr: usize,
        sr: usize,
    ) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        num_groups: usize,
        n: usize,
        k: usize,
        nr: usize,
        kr: usize,
        sr: usize,
        rhs: *const i8,
        bias: *const f32,
        scale: *const f32,
        rhs_packed: *mut c_void,
        extra_bytes: usize,
        params: *const RhsPackParams,
    );
    fn kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
        m: usize,
        k: usize,
        mr: usize,
        kr: usize,
        sr: usize,
    ) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn kai_run_lhs_quant_pack_qai8dxp_f32(
        m: usize,
        k: usize,
        mr: usize,
        kr: usize,
        sr: usize,
        m_idx_start: usize,
        lhs: *const f32,
        lhs_stride: usize,
        lhs_packed: *mut c_void,
    );
    #[allow(clippy::too_many_arguments)]
    fn kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
        m: usize,
        n: usize,
        k: usize,
        lhs_packed: *const c_void,
        rhs_packed: *const c_void,
        dst: *mut f32,
        dst_stride_row: usize,
        dst_stride_col: usize,
        scalar_min: f32,
        scalar_max: f32,
    );
}

/// weights packed for the KleidiAI kernel (int8 + per-channel scales + bias)
pub struct KaiRhs {
    pub n: usize,
    pub k: usize,
    data: AVec<u8>,
}

impl KaiRhs {
    /// `q`: int8 weights [n, k] row-major, `scales` [n], `bias` [n] (zeros when None)
    pub fn pack(q: &[i8], n: usize, k: usize, scales: &[f32], bias: Option<&[f32]>) -> KaiRhs {
        let zeros = vec![0f32; n];
        let b = bias.unwrap_or(&zeros);
        // SAFETY: sizes come from the library's own size function; inputs hold n*k / n elements.
        unsafe {
            let sz = kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(n, k, NR, KR, SR);
            let mut data = AVec::zeroed(sz);
            let params = RhsPackParams {
                lhs_zero_point: 1,
                scale_multiplier: 1.0,
            };
            kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
                1,
                n,
                k,
                NR,
                KR,
                SR,
                q.as_ptr(),
                b.as_ptr(),
                scales.as_ptr(),
                data.as_mut_slice().as_mut_ptr() as *mut c_void,
                0,
                &params,
            );
            KaiRhs { n, k, data }
        }
    }

    pub fn bytes(&self) -> usize {
        self.data.len()
    }
}

/// per-session buffers of the KleidiAI path
#[derive(Default)]
pub struct KaiScratch {
    lhs: AVec<u8>,
    pre: Vec<f32>,
    out: Vec<f32>,
}

/// y[m, ldy] = epi(x[m, k]·Wᵀ + b) through KleidiAI; `pre`: smoothing factors multiplied into x first
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    w: &KaiRhs,
    x: &[f32],
    m: usize,
    pre: Option<&[f32]>,
    y: &mut [f32],
    ldy: usize,
    epi: Epi,
    s: &mut KaiScratch,
) {
    let (n, k) = (w.n, w.k);
    let src: &[f32] = match pre {
        Some(p) => {
            s.pre.resize(m * k, 0.0);
            for r in 0..m {
                for j in 0..k {
                    s.pre[r * k + j] = x[r * k + j] * p[j];
                }
            }
            &s.pre
        }
        None => &x[..m * k],
    };
    // SAFETY: buffer sizes from the library's size functions; dst rows have stride ldy >= n floats.
    unsafe {
        let lsz = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, MR, KR, SR);
        s.lhs.resize(lsz);
        let lp = s.lhs.as_mut_slice().as_mut_ptr() as *mut c_void;
        kai_run_lhs_quant_pack_qai8dxp_f32(m, k, MR, KR, SR, 0, src.as_ptr(), k * 4, lp);
        let direct = !epi.accumulate && epi.alpha == 1.0;
        if direct {
            kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
                m,
                n,
                k,
                lp,
                w.data.as_slice().as_ptr() as *const c_void,
                y.as_mut_ptr(),
                ldy * 4,
                4,
                f32::MIN,
                f32::MAX,
            );
        } else {
            s.out.resize(m * n, 0.0);
            kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
                m,
                n,
                k,
                lp,
                w.data.as_slice().as_ptr() as *const c_void,
                s.out.as_mut_ptr(),
                n * 4,
                4,
                f32::MIN,
                f32::MAX,
            );
            for r in 0..m {
                let yr = &mut y[r * ldy..r * ldy + n];
                let o = &s.out[r * n..(r + 1) * n];
                if epi.accumulate {
                    for c in 0..n {
                        yr[c] += epi.alpha * o[c];
                    }
                } else {
                    for c in 0..n {
                        yr[c] = epi.alpha * o[c];
                    }
                }
            }
        }
    }
}
