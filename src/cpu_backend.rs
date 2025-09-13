use crate::tensor::{DType, DeviceBuffer};
use anyhow::Result;
use ndarray::ArrayD;
use std::sync::Arc;

pub struct CpuBuffer {
    data: Arc<ArrayD<f32>>,
}

impl CpuBuffer {
    pub fn from_array(a: Arc<ArrayD<f32>>) -> Self {
        Self { data: a }
    }
}

impl DeviceBuffer for CpuBuffer {
    fn dtype(&self) -> DType {
        DType::F32
    }
    fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }
    fn to_host_f32(&self) -> anyhow::Result<Vec<f32>> {
        // Fast-path: if the ndarray storage is contiguous we can take a
        // direct slice and clone that into a Vec in one memcpy-style op.
        if let Some(slice) = self.data.as_slice_memory_order() {
            Ok(slice.to_vec())
        } else {
            let mut v = Vec::with_capacity(self.data.len());
            v.extend(self.data.iter().cloned());
            Ok(v)
        }
    }
    fn box_clone(&self) -> Box<dyn DeviceBuffer> {
        Box::new(CpuBuffer {
            data: self.data.clone(),
        })
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub fn buffer_from_array_cpu(a: Arc<ArrayD<f32>>) -> Result<Box<dyn DeviceBuffer>> {
    Ok(Box::new(CpuBuffer::from_array(a)))
}
