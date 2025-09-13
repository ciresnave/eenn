use eenn::{FunctionRegistry, Neuron, Stage, bias, relu, scale, sigmoid};

#[test]
fn neuron_forward_computes_pipeline() {
    let mut registry = FunctionRegistry::empty();
    registry.register_fn("relu", relu, "Rectified Linear Unit");
    registry.register_fn("sigmoid", sigmoid, "Logistic sigmoid");

    let stages: Vec<Stage> = vec![
        Stage::new(scale(0.5)),
        Stage::new(bias(0.25)),
        Stage::from_arc(registry.get("relu").expect("missing relu")),
    ];
    let output = Stage::from_arc(registry.get("sigmoid").expect("missing sigmoid"));
    let neuron = Neuron::new(stages, output);

    let x = -1.0f32;
    // apply scale(0.5) => -0.5; bias(0.25) => -0.25; relu => 0.0; sigmoid(0.0) => 0.5
    let y = neuron.forward(x);
    assert!((y - 0.5).abs() < 1e-6);
}

#[test]
fn function_registry_get_missing_returns_none() {
    let registry = FunctionRegistry::empty();
    assert!(registry.get("not-there").is_none());
}

#[test]
fn register_closure_factory_and_use() {
    let mut registry = FunctionRegistry::empty();
    // register a stateful closure produced by scale(0.75)
    registry.register("scale_075", scale(0.75), "Scale by 0.75");

    let f_arc = registry.get("scale_075").expect("missing scale_075");
    // call the function via the Arc
    let out = (f_arc)(2.0f32);
    assert!((out - 1.5).abs() < 1e-6);
}

#[test]
fn remove_and_replace_behaviour() {
    let mut registry = FunctionRegistry::empty();
    registry.register_fn("relu", relu, "relu");
    assert!(registry.get("relu").is_some());

    // remove
    assert!(registry.remove("relu"));
    assert!(registry.get("relu").is_none());

    // replace (insert)
    let prev = registry.replace("relu", relu, "relu v2");
    assert!(prev.is_none());
    assert!(registry.get("relu").is_some());

    // replace again (should return previous)
    let prev2 = registry.replace("relu", scale(2.0), "scale");
    assert!(prev2.is_some());
}
