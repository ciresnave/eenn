#![cfg(feature = "rkyv")]

use eenn::models::{NeuronDef, StageDef};
use eenn::{FunctionRegistry, relu};

#[test]
fn archived_borrow_str_and_slice() {
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

    // Serialize using rkyv
    let bytes = def.to_bytes().expect("serialize");

    // Rehydrate directly from the archived view without allocating owned NeuronDef
    let neuron = NeuronDef::from_bytes_zero_copy_to_neuron(&bytes, &reg).expect("rehydrate");

    // Smoke check the neuron works
    let out = neuron.forward(2.0);
    // relu( scale( relu(2.0) ) ) => relu( scale(2.0) ) => relu(1.0) => 1.0
    assert_eq!(out, 1.0);
}
