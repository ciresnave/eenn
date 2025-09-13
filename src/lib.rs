pub mod function_registry;
pub use function_registry::{FuncMeta, FunctionRegistry};
pub mod fingerprint;
pub use fingerprint::{fingerprint_sha256_hex, to_canonical_json};
pub mod observability;
pub use observability::InMemoryProfiler;
pub use observability::{CacheStats, ExecutionContext, Profiler};
pub mod kernel_cache;
pub use kernel_cache::KernelCache;
pub mod partitioner;
pub use partitioner::{GreedyPartitioner, Partitioner, Segment};
pub mod calibration;
pub use calibration::Calibration;
pub mod signature;
pub use signature::{InMemoryTrustStore, TrustStore, verify_manifest};
#[cfg(feature = "tokio")]
pub mod precompile_worker;
#[cfg(feature = "tokio")]
pub use precompile_worker::{
    CancelHandle, CancellationToken, CompileFn, CompileTask, PrecompileService,
    spawn_precompile_worker,
};
#[cfg(feature = "tokio")]
pub use tokio::sync::Notify;
#[cfg(not(feature = "gpu"))]
pub mod cpu_backend;
#[cfg(feature = "gpu")]
pub mod gpu_backend;
pub mod models;
pub mod node;
pub mod tensor;
pub use node::{ConcatNode, MergeNode, ReshapeNode};
pub use node::{Node, Sequential, StageNode, TickCtx};
use std::sync::Arc;
pub use tensor::{DType, PortId, PortSpec, PortTensors, Shape, Tensor};

// ---------- stateless ops (ordinary fn pointers) ----------
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
pub fn tanh_f(x: f32) -> f32 {
    x.tanh()
}

// ---------- stateful factories (closures that capture params) ----------
pub fn scale(w: f32) -> impl Fn(f32) -> f32 + Send + Sync + 'static {
    move |x| w * x
}
pub fn bias(b: f32) -> impl Fn(f32) -> f32 + Send + Sync + 'static {
    move |x| x + b
}

#[derive(Clone)]
pub struct Stage(Arc<dyn Fn(f32) -> f32 + Send + Sync>);

impl Stage {
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(f32) -> f32 + Send + Sync + 'static,
    {
        Stage(Arc::new(f))
    }

    /// Create a Stage from an existing Arc<dyn Fn(...)> (e.g. from the registry).
    pub fn from_arc(f: Arc<dyn Fn(f32) -> f32 + Send + Sync + 'static>) -> Self {
        Stage(f)
    }

    pub fn call(&self, x: f32) -> f32 {
        (self.0)(x)
    }
}

pub struct Neuron {
    stages: Vec<Stage>, // applied in order
    output: Stage,      // final op
}

impl Neuron {
    pub fn new(stages: Vec<Stage>, output: Stage) -> Self {
        Self { stages, output }
    }

    pub fn forward(&self, mut x: f32) -> f32 {
        for s in &self.stages {
            x = s.call(x);
        }
        self.output.call(x)
    }
}
// Library-only: examples and binaries should live in `src/main.rs` or `src/bin/`.
