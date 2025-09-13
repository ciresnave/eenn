// Small common helpers intended to be shared between CPU helpers and rust-gpu
// shader front-ends. Keep this minimal: simple shape helpers and dtype helpers.

pub fn num_elements(dims: &[usize]) -> usize {
    dims.iter().product()
}
