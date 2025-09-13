use criterion::{Criterion, criterion_group, criterion_main};

fn tiny_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiny");
    group.bench_function("noop", |b| b.iter(|| ()));
    group.finish();
}

criterion_group!(benches, tiny_bench);
criterion_main!(benches);
