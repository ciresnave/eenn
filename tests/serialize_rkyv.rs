#![cfg(feature = "rkyv")]

use eenn::FunctionRegistry;
use eenn::models::{NeuronDef, StageDef};
use std::sync::Arc;

#[test]
fn rkyv_roundtrip_and_rehydrate() {
    let mut registry = FunctionRegistry::empty();
    registry.register_fn("relu", eenn::relu, "ReLU");
    registry.register_factory("scale", |params: &[f32]| {
        // scale(...) returns an impl Fn; wrap in Arc to satisfy factory return type
        Arc::new(eenn::scale(params[0])) as Arc<dyn Fn(f32) -> f32 + Send + Sync + 'static>
    });

    // Build a neuron def: scale(2.0) -> relu
    let stages = vec![StageDef::Factory {
        name: "scale".to_string(),
        params: vec![2.0],
    }];
    let output = StageDef::Named("relu".to_string());
    let def = NeuronDef::new(stages, output);

    let bytes = def.to_bytes().expect("serialize");
    let def2 = NeuronDef::from_bytes(&bytes).expect("deserialize");

    let neuron = def2.to_neuron(&registry).expect("rehydrate");
    let out = neuron.forward(1.5);
    assert_eq!(out, eenn::relu(2.0 * 1.5));
}
