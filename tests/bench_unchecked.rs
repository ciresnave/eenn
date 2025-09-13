#![cfg(all(feature = "rkyv", feature = "rkyv_unchecked"))]

use eenn::models::{NeuronDef, StageDef};
use eenn::{FunctionRegistry, relu};
use std::time::Instant;

#[test]
fn bench_safe_vs_unchecked_rehydrate() {
    let mut reg = FunctionRegistry::empty();
    reg.register_fn("relu", relu, "ReLU");
    reg.register_factory("scale", |params: &[f32]| {
        let p = params.to_vec();
        std::sync::Arc::new(move |x| p[0] * x)
    });

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

    let bytes = def.to_bytes().expect("serialize");

    // warm-up
    let _ = NeuronDef::from_bytes_zero_copy_to_neuron(&bytes, &reg).expect("rehydrate");

    let start = Instant::now();
    for _ in 0..1000 {
        let _ = NeuronDef::from_bytes_zero_copy_to_neuron(&bytes, &reg).expect("rehydrate");
    }
    let safe_ms = start.elapsed().as_millis();

    // unsafe path
    let start = Instant::now();
    for _ in 0..1000 {
        let _ =
            unsafe { NeuronDef::from_bytes_unchecked_to_neuron(&bytes, &reg).expect("rehydrate") };
    }
    let unchecked_ms = start.elapsed().as_millis();

    println!("safe(ms): {} unchecked(ms): {}", safe_ms, unchecked_ms);

    // sanity check: results should be identical
    let safe_n = NeuronDef::from_bytes_zero_copy_to_neuron(&bytes, &reg).expect("rehydrate");
    let unchecked_n =
        unsafe { NeuronDef::from_bytes_unchecked_to_neuron(&bytes, &reg).expect("rehydrate") };
    assert_eq!(safe_n.forward(2.0), unchecked_n.forward(2.0));
}
