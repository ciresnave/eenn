use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use eenn::partitioner::Op;
use eenn::{GreedyPartitioner, Partitioner};
use std::sync::Arc;

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
        // alternate gpu-capable ops with small compute and occasional large transfer
        let gpu = (i % 5) != 4; // every 5th is non-gpu
        let c = if gpu { 1.0 } else { 0.0 };
        let t = if gpu && (i % 7 == 0) { 200_000 } else { 0 };
        v.push(Arc::new(SyntheticOp { gpu, c, t }) as Arc<dyn Op>);
    }
    v
}

fn bench_partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("partitioner_lookahead");
    let sizes = [50usize, 200usize, 800usize];
    let lookahead_vals = [0usize, 1usize, 2usize, 4usize, 8usize];

    for &n in &sizes {
        let ops = make_ops(n);
        group.throughput(Throughput::Elements(n as u64));
        for &look in &lookahead_vals {
            let p = GreedyPartitioner {
                lookahead: look,
                bytes_per_ms: 1_000_000.0,
                max_ops_in_fusion: 512,
            };
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("n{}_look{}", n, look)),
                &ops,
                |b, ops| {
                    b.iter(|| {
                        let _ = p.partition(ops);
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_partition);
criterion_main!(benches);
