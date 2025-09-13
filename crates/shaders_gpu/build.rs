fn main() {
    // spirv-builder will compile the Rust shader crate to SPIR-V when the
    // environment and toolchain are set up. Keep build.rs minimal and
    // tolerant: if spirv-builder is not available or the target isn't
    // configured, we simply skip producing SPIR-V silently.
    if std::env::var("SKIP_SPIRV_BUILD").is_ok() {
        println!("cargo:warning=skipping spirv build (SKIP_SPIRV_BUILD set)");
        return;
    }

    match spirv_builder::Builder::new("shaders_gpu", "spirv-unknown-unknown")
        .release()
        .print_metadata(false)
        .try_build()
    {
        Ok(_out) => {
            // spirv-builder places artifacts under OUT_DIR; export a path so
            // host code in the workspace can locate the generated .spv.
            if let Ok(out_dir) = std::env::var("OUT_DIR") {
                let spv = std::path::Path::new(&out_dir).join("shaders_gpu.spv");
                println!("cargo:rustc-env=SPIRV_ARTIFACT={}", spv.display());
            }
            println!("cargo:rerun-if-changed=src/shaders.rs");
        }
        Err(e) => {
            println!("cargo:warning=spirv build failed: {}", e);
        }
    }
}
