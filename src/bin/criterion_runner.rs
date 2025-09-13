use criterion::Criterion;
use eenn::partitioner::Op;
use eenn::{GreedyPartitioner, Partitioner};
use std::sync::Arc;
// Use core_affinity for a portable affinity helper
fn pin_to_core(core: usize) {
    // no-op for now (platform-specific affinity not enabled in Cargo.toml)
    let _ = core;
}

struct SyntheticOp {
    gpu: bool,
    c: f64,
    t: usize,
}
impl Op for SyntheticOp {
    fn name(&self) -> &str {
        "op"
    }
    fn gpu_capable(&self) -> bool {
        self.gpu
    }
    fn estimate(&self) -> (f64, usize) {
        (self.c, self.t)
    }
}

fn make_ops(n: usize) -> Vec<Arc<dyn Op>> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let gpu = (i % 5) != 4;
        let c = if gpu { 1.0 } else { 0.0 };
        let t = if gpu && (i % 7 == 0) { 200_000 } else { 0 };
        v.push(Arc::new(SyntheticOp { gpu, c, t }) as Arc<dyn Op>);
    }
    v
}

fn main() {
    // Build a Criterion instance programmatically, pointing to target/criterion
    let mut crit = Criterion::default();

    // Focused run: only n=800, look=8 to collect instrumentation for the worst-case observed
    let n = 800usize;
    let look = 8usize;
    let ops = make_ops(n);
    let p = GreedyPartitioner {
        lookahead: look,
        bytes_per_ms: 1_000_000.0,
        max_ops_in_fusion: 512,
    };
    let id = format!("n{}_look{}", n, look);

    // Pin to CPU 0 to reduce scheduling noise (Windows-only effective)
    pin_to_core(0);

    // Increase Criterion sample size and measurement time for this expensive benchmark
    let mut group = crit.benchmark_group("partitioner_focused");
    group.measurement_time(std::time::Duration::from_secs(10));
    group.sample_size(200);
    group.bench_function(&id, |b| {
        b.iter(|| {
            let _ = p.partition(&ops);
        })
    });
    group.finish();

    // Finalize and write reports
    crit.final_summary();
    // Dump partitioner instrumentation (counts/histograms)
    // Dump both the global and per-benchmark instrumentation
    eenn::partitioner::dump_instrumentation();
    eenn::partitioner::dump_instrumentation_for(&id);
}
