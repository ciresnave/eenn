// Front-end shader crate: exposes CPU-side helpers that will be referenced by the shader
// crate (when compiling to spirv) so that CPU and GPU implementations can share logic.

pub fn canonical_layout_dims(dims: &[usize]) -> Vec<u32> {
    dims.iter().map(|d| *d as u32).collect()
}
