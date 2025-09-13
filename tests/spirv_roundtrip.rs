#![cfg(feature = "gpu")]

use eenn::{gpu_backend, tensor};
use std::fs;
use std::path::PathBuf;
// Arc not needed in this test currently

#[test]
fn spirv_roundtrip_copy_if_available() {
    // Attempt to discover SPIR-V artifact produced by the shader crate. The
    // shader build exports SPIRV_ARTIFACT env var when available. If missing
    // we'll skip the test.
    let maybe = std::env::var("SPIRV_ARTIFACT");
    if maybe.is_err() {
        eprintln!("SPIR-V artifact not found, skipping spirv_roundtrip test");
        return;
    }
    let spv = PathBuf::from(maybe.unwrap());
    if !spv.exists() {
        eprintln!("SPIR-V artifact {} missing, skipping", spv.display());
        return;
    }

    // Load the SPIR-V as u32 words and dispatch copy.
    let bytes = fs::read(&spv).expect("read spv");
    if bytes.len() % 4 != 0 {
        eprintln!("invalid spv length, skipping");
        return;
    }
    let words: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&bytes).to_vec();

    // Build a small 1-D tensor and run host->device -> kernel -> host roundtrip
    let a = tensor::Tensor::vector_f32(vec![1.0f32, 2.0, 3.0]);
    let dev = a.host_to_device().expect("host_to_device");
    let (dev_buf, _shape) = match &dev.storage {
        tensor::TensorStorage::Device(b) => (b, dev.shape.clone()),
        _ => panic!("not a device buffer"),
    };

    // Create a destination buffer of the same shape
    let zero = tensor::Tensor::vector_f32(vec![0.0f32; 3]);
    let dst = zero.host_to_device().expect("host_to_device dst");
    let dst_buf = match &dst.storage {
        tensor::TensorStorage::Device(b) => b,
        _ => panic!("not a device buffer"),
    };

    // Downcast DeviceBuffer to access raw wgpu buffer and device.
    // We only support the GpuBuffer implementation here.
    let src_g = dev_buf
        .as_any()
        .downcast_ref::<gpu_backend::GpuBuffer>()
        .expect("src is GpuBuffer");
    let dst_g = dst_buf
        .as_any()
        .downcast_ref::<gpu_backend::GpuBuffer>()
        .expect("dst is GpuBuffer");

    let device = src_g.device.clone();
    // Dispatch
    device
        .dispatch_spirv_copy(&words, src_g.raw_buffer(), dst_g.raw_buffer(), 3)
        .expect("dispatch spv");

    // Read back destination
    let out = device
        .read_buffer_to_host(dst_g.raw_buffer(), 3)
        .expect("read back");
    assert_eq!(out, vec![1.0f32, 2.0, 3.0]);
}
