use eenn::FunctionRegistry;
use eenn::models::{NeuronDef, StageDef};

#[test]
fn neuron_serialization_roundtrip_and_rehydrate() {
    let mut reg = FunctionRegistry::empty();
    // Register some named stateless functions
    reg.register_fn("relu", eenn::relu, "ReLU");

    // Register a factory that creates a linear scale+bias closure from params
    reg.register_factory("linear", |params: &[f32]| {
        let w = params.first().cloned().unwrap_or(1.0);
        let b = params.get(1).cloned().unwrap_or(0.0);
        std::sync::Arc::new(move |x: f32| w * x + b)
    });

    // Build a NeuronDef that uses a named op and a factory
    let def = NeuronDef::new(
        vec![
            StageDef::Named("relu".to_string()),
            StageDef::Factory {
                name: "linear".to_string(),
                params: vec![2.0, 1.0],
            },
        ],
        StageDef::Bias { b: 0.5 },
    );

    // Serialize -> deserialize
    let bytes = def.to_bytes().expect("serialize");
    let def2 = NeuronDef::from_bytes(&bytes).expect("deserialize");

    // Rehydrate and run
    let neuron = def2.to_neuron(&reg).expect("rehydrate");
    let out = neuron.forward(-1.0);

    // relu(-1.0) = 0; factory linear with w=2,b=1 => 1; bias 0.5 => 1.5
    assert!((out - 1.5).abs() < 1e-6, "unexpected output: {}", out);
}
