// Placeholder shader crate for rust-gpu. This crate is a scaffold: when you
// want to compile shaders to SPIR-V with rust-gpu, add a build.rs that uses
// `spirv-builder` and convert functions to #[spirv(compute)] entry points.

pub use shaders_common::num_elements;
pub use shaders_front::canonical_layout_dims;

// CPU fallback
pub fn copy_flat(src: &[f32], dst: &mut [f32]) {
    dst.copy_from_slice(src);
}

// rust-gpu shader entrypoint (only compiled when the crate is built as a
// shader via spirv-builder with the `spirv` cfg). This is a minimal copy
// kernel that copies `src[i]` to `dst[i]`.
#[cfg(spirv)]
pub mod shader {
    use spirv_std::spirv;

    #[spirv(compute(threads(64)))]
    pub fn copy_kernel(
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] src: &[f32],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dst: &mut [f32],
    ) {
        let idx = spirv_std::compute_index_in_global_invocation();
        if (idx as usize) < src.len() && (idx as usize) < dst.len() {
            dst[idx as usize] = src[idx as usize];
        }
    }
}
