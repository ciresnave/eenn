#![cfg(all(feature = "rkyv", feature = "rkyv_unchecked"))]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use eenn::models::{NeuronDef, StageDef};
use eenn::{FunctionRegistry, relu};

fn make_registry() -> FunctionRegistry {
    let mut reg = FunctionRegistry::empty();
    reg.register_fn("relu", relu, "ReLU");
    reg.register_factory("scale", |params: &[f32]| {
        let p = params.to_vec();
        std::sync::Arc::new(move |x| p[0] * x)
    });
    reg
}

fn make_bytes() -> Vec<u8> {
    let def = NeuronDef::new(
        vec![
            StageDef::Named("relu".to_string()),
            StageDef::Factory {
                name: "scale".to_string(),
                params: vec![0.5],
            },
        ],
        StageDef::Named("relu".to_string()),
    );
    def.to_bytes().expect("serialize")
}

fn bench_rehydrate(c: &mut Criterion) {
    let reg = make_registry();
    let bytes = make_bytes();

    let mut group = c.benchmark_group("rehydration");
    group.throughput(Throughput::Elements(1));

    group.bench_function(BenchmarkId::new("safe", "rehydrate"), |b| {
        b.iter(|| {
            let _ = NeuronDef::from_bytes_zero_copy_to_neuron(&bytes, &reg).expect("rehydrate");
        });
    });

    group.bench_function(BenchmarkId::new("unchecked", "rehydrate"), |b| {
        b.iter(|| unsafe {
            NeuronDef::from_bytes_unchecked_to_neuron(&bytes, &reg).expect("rehydrate")
        });
    });

    group.finish();
}

criterion_group!(benches, bench_rehydrate);
criterion_main!(benches);
