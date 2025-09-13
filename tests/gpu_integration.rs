#![cfg(feature = "gpu")]

use eenn::tensor::Tensor;

#[test]
fn gpu_roundtrip_matches_cpu_ndarray() {
    // Create a 2x3 tensor on CPU
    let cpu = Tensor::matrix_f32(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    // Move to device
    let dev = cpu.host_to_device().expect("host_to_device");
    // Move back
    let back = dev.device_to_host().expect("device_to_host");
    let a = cpu.to_ndarray_f32().unwrap();
    let b = back.to_ndarray_f32().unwrap();
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.as_slice_memory_order(), b.as_slice_memory_order());
}
