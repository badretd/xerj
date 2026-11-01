// Optional KleidiAI micro-kernels (feature "kleidiai", aarch64 only): compiled from the vendored sources in third_party/kleidiai.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(xerj_kleidiai)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=third_party/kleidiai");
    #[cfg(feature = "kleidiai")]
    {
        if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64") {
            let base = "third_party/kleidiai";
            let m = format!("{base}/kai/ukernels/matmul");
            cc::Build::new()
                .include(base)
                .file(format!("{m}/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c"))
                .file(format!("{m}/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm_asm.S"))
                .file(format!("{m}/pack/kai_lhs_quant_pack_qai8dxp_f32.c"))
                .file(format!("{m}/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c"))
                // the micro-kernel source requires __ARM_FEATURE_MATMUL_INT8; it is only called after a run-time FEAT_I8MM check
                .flag("-march=armv8.6-a+i8mm+dotprod")
                .opt_level(3)
                .warnings(false)
                .compile("xerj_kleidiai");
            println!("cargo:rustc-cfg=xerj_kleidiai");
        }
    }
}
