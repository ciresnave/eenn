// rust-gpu shader source. When compiled with `spirv-builder` this file will be
// turned into SPIR-V. Keep the code minimal and use the rust-gpu attributes
// when enabled. This file is also valid Rust for host-side builds (not used at
// runtime on the host) but must be `#[cfg(spirv)]` for real shader functions.

// The real rust-gpu crate would use `#[spirv(compute(threads(64)))]` and the
// rust-gpu prelude. Here we include a textual placeholder so builds that don't
// have rust-gpu toolchain available still succeed.

#[cfg(not(spirv))]
pub fn _placeholder_shader_copy(_src_ptr: *const f32, _dst_ptr: *mut f32, _len: u32) {
    // placeholder for rust-gpu compiled entrypoint
}

#[cfg(spirv)]
// rust-gpu entrypoints would go here; example signature below:
// #[spirv(compute(threads(64)))]
// pub fn copy_kernel(#[spirv(storage_buffer, descriptor_set = 0, binding = 0)] src: &[f32],
//                    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dst: &mut [f32]) {
//     // body: dst[i] = src[i]
// }
