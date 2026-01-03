fn main() {
    // Detect CPU features at build time and emit custom cfg flags.
    // This allows us to use cfg(pixelflow_avx512f) instead of cfg(target_feature = "avx512f"),
    // which doesn't work with target-cpu=native.

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("cargo:rustc-cfg=pixelflow_avx512f");
        }
        if is_x86_feature_detected!("avx2") {
            println!("cargo:rustc-cfg=pixelflow_avx2");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // ARM always has NEON on aarch64
        println!("cargo:rustc-cfg=pixelflow_neon");
    }
}
