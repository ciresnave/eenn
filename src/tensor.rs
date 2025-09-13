use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape(pub Vec<Option<usize>>);
impl Shape {
    pub fn scalar() -> Self {
        Self(vec![])
    }
    pub fn vector(n: usize) -> Self {
        Self(vec![Some(n)])
    }

    /// Return total number of elements if all dims are concrete (no `None`).
    pub fn num_elements(&self) -> Option<usize> {
        let mut acc: usize = 1;
        for d in &self.0 {
            match d {
                Some(n) => acc = acc.saturating_mul(*n),
                None => return None,
            }
        }
        Some(acc)
    }

    /// Check if this shape matches `other`, allowing optional broadcasting from
    /// scalar to a 1-D vector when `allow_broadcast` is true.
    pub fn matches(&self, other: &Shape, allow_broadcast: bool) -> bool {
        if self == other {
            return true;
        }
        if allow_broadcast {
            // scalar can broadcast to any single-dim vector
            if self.0.is_empty() && other.0.len() == 1 {
                return true;
            }
        }
        false
    }
}

pub enum TensorStorage {
    NdF32(Arc<ArrayD<f32>>),
    NdF64(Arc<ArrayD<f64>>),
    Device(Box<dyn DeviceBuffer>),
}

impl Clone for TensorStorage {
    fn clone(&self) -> Self {
        match self {
            TensorStorage::NdF32(a) => TensorStorage::NdF32(a.clone()),
            TensorStorage::NdF64(a) => TensorStorage::NdF64(a.clone()),
            TensorStorage::Device(d) => TensorStorage::Device(d.box_clone()),
        }
    }
}

impl std::fmt::Debug for TensorStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorStorage::NdF32(a) => f.debug_tuple("NdF32").field(a).finish(),
            TensorStorage::NdF64(a) => f.debug_tuple("NdF64").field(a).finish(),
            TensorStorage::Device(_) => f.debug_tuple("Device").finish(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub dtype: DType,
    pub shape: Shape,
    pub storage: TensorStorage,
}

impl Tensor {
    pub fn scalar_f32(v: f32) -> Self {
        // Represent scalar as 0-d ndarray
        let arr = ArrayD::from_shape_vec(IxDyn(&[]), vec![v]).expect("scalar shape");
        Self {
            dtype: DType::F32,
            shape: Shape::scalar(),
            storage: TensorStorage::NdF32(Arc::new(arr)),
        }
    }

    pub fn vector_f32(v: Vec<f32>) -> Self {
        let len = v.len();
        let arr = ArrayD::from_shape_vec(IxDyn(&[len]), v).expect("vector shape");
        Self {
            dtype: DType::F32,
            shape: Shape::vector(len),
            storage: TensorStorage::NdF32(Arc::new(arr)),
        }
    }

    /// Construct an ND array-backed tensor from raw data and explicit dims.
    pub fn from_vec_nd_f32(data: Vec<f32>, dims: Vec<usize>) -> anyhow::Result<Self> {
        let expected: usize = dims.iter().product();
        if data.len() != expected {
            return Err(anyhow::anyhow!("data length does not match dims"));
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&dims), data)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let shape = Shape(dims.into_iter().map(Some).collect());
        Ok(Self {
            dtype: DType::F32,
            shape,
            storage: TensorStorage::NdF32(Arc::new(arr)),
        })
    }

    /// Construct matrix helper (2-D) from flat Vec
    pub fn matrix_f32(rows: usize, cols: usize, data: Vec<f32>) -> anyhow::Result<Self> {
        Self::from_vec_nd_f32(data, vec![rows, cols])
    }

    pub fn as_f32_scalar(&self) -> Option<f32> {
        if self.shape.0.is_empty() {
            match &self.storage {
                TensorStorage::NdF32(a) => {
                    if a.ndim() == 0 {
                        a.first().cloned()
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            None
        }
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match &self.storage {
            TensorStorage::NdF32(a) => a.as_slice_memory_order(),
            _ => None,
        }
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.0.is_empty()
    }
    pub fn is_vector(&self) -> bool {
        self.shape.0.len() == 1
    }

    /// Try to get an ndarray view for f32 storage (cloning Vec into Array if necessary).
    pub fn to_ndarray_f32(&self) -> Option<Arc<ArrayD<f32>>> {
        match &self.storage {
            TensorStorage::NdF32(a) => Some(a.clone()),
            TensorStorage::Device(dev) => {
                if dev.dtype() == DType::F32
                    && let (Ok(data), Some(dims)) = (
                        dev.to_host_f32(),
                        self.shape.0.iter().copied().collect::<Option<Vec<usize>>>(),
                    )
                    && let Ok(arr) = ArrayD::from_shape_vec(IxDyn(&dims), data)
                {
                    return Some(Arc::new(arr));
                }
                None
            }
            _ => None,
        }
    }
}

/// Trait representing a device buffer. Minimal methods for the placeholder.
pub trait DeviceBuffer: Send + Sync {
    fn dtype(&self) -> DType;
    fn shape(&self) -> Vec<usize>;
    fn to_host_f32(&self) -> anyhow::Result<Vec<f32>>;
    /// Allow downcasting from trait object.
    fn as_any(&self) -> &dyn std::any::Any;
    /// Return a boxed clone for trait object cloning.
    fn box_clone(&self) -> Box<dyn DeviceBuffer>;
}

// CPU-backed DeviceBuffer is provided by `cpu_backend` (a small module).
// When the `gpu` feature is disabled we use that implementation so the
// same DeviceBuffer trait is available for host-only tests and execution.

impl Tensor {
    /// Move tensor to a device buffer (dummy implementation).
    pub fn host_to_device(&self) -> anyhow::Result<Self> {
        match &self.storage {
            TensorStorage::NdF32(a) => {
                #[cfg(feature = "gpu")]
                {
                    let dev = crate::gpu_backend::buffer_from_array(a.clone())?;
                    Ok(Self {
                        dtype: self.dtype,
                        shape: self.shape.clone(),
                        storage: TensorStorage::Device(dev),
                    })
                }
                #[cfg(not(feature = "gpu"))]
                {
                    let dev = crate::cpu_backend::buffer_from_array_cpu(a.clone())?;
                    Ok(Self {
                        dtype: self.dtype,
                        shape: self.shape.clone(),
                        storage: TensorStorage::Device(dev),
                    })
                }
            }
            _ => Err(anyhow::anyhow!(
                "host_to_device: only NdF32 supported in dummy device"
            )),
        }
    }

    /// Fetch device tensor back to host (dummy implementation).
    pub fn device_to_host(&self) -> anyhow::Result<Self> {
        match &self.storage {
            TensorStorage::Device(dev) => {
                if dev.dtype() == DType::F32 {
                    let data = dev.to_host_f32()?;
                    if let Some(dims) = self.shape.0.iter().copied().collect::<Option<Vec<usize>>>()
                    {
                        let arr = ArrayD::from_shape_vec(IxDyn(&dims), data)
                            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                        return Ok(Self {
                            dtype: DType::F32,
                            shape: self.shape.clone(),
                            storage: TensorStorage::NdF32(Arc::new(arr)),
                        });
                    }
                }
                Err(anyhow::anyhow!("device_to_host: failed"))
            }
            _ => Err(anyhow::anyhow!("device_to_host: not a device tensor")),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PortId(pub &'static str);

#[derive(Clone, Debug)]
pub enum Arity {
    Exactly(usize),
    Range { min: usize, max: Option<usize> },
}

#[derive(Clone, Debug)]
pub struct PortSpec {
    pub id: PortId,
    pub arity: Arity,
    pub dtype: Option<DType>,
    pub shape: Option<Shape>,
    pub allow_broadcast: bool,
}

pub type PortTensors = HashMap<PortId, Vec<Tensor>>;

impl PortId {
    pub fn as_str(&self) -> &'static str {
        self.0
    }
}

impl PortSpec {
    /// Validate a slice of tensors against this PortSpec.
    pub fn validate_tensors(&self, tensors: &[Tensor]) -> Result<(), String> {
        // arity check
        match &self.arity {
            Arity::Exactly(n) => {
                if tensors.len() != *n {
                    return Err(format!(
                        "port {} expected {} tensors, got {}",
                        self.id.as_str(),
                        n,
                        tensors.len()
                    ));
                }
            }
            Arity::Range { min, max } => {
                if tensors.len() < *min {
                    return Err(format!(
                        "port {} expected at least {} tensors, got {}",
                        self.id.as_str(),
                        min,
                        tensors.len()
                    ));
                }
                if let Some(maxv) = max
                    && tensors.len() > *maxv
                {
                    return Err(format!(
                        "port {} expected at most {} tensors, got {}",
                        self.id.as_str(),
                        maxv,
                        tensors.len()
                    ));
                }
            }
        }

        for t in tensors.iter() {
            if let Some(dtype) = &self.dtype
                && &t.dtype != dtype
            {
                return Err(format!(
                    "port {} expected dtype {:?}, got {:?}",
                    self.id.as_str(),
                    dtype,
                    t.dtype
                ));
            }
            if let Some(spec_shape) = &self.shape
                && !t.shape.matches(spec_shape, self.allow_broadcast)
            {
                return Err(format!(
                    "port {} tensor shape mismatch: expected {:?}, got {:?}",
                    self.id.as_str(),
                    spec_shape,
                    t.shape
                ));
            }
        }

        Ok(())
    }
}

/// Validate a set of PortSpecs against provided PortTensors. Returns Ok or the first error.
pub fn validate_port_tensors(specs: &[PortSpec], tensors: &PortTensors) -> Result<(), String> {
    for spec in specs.iter() {
        let v = tensors.get(&spec.id).cloned().unwrap_or_default();
        spec.validate_tensors(&v)?;
    }
    Ok(())
}
