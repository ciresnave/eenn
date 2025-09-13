use eenn::{FuncMeta, FunctionRegistry, Neuron, Stage, bias, relu, scale, sigmoid};
use std::collections::HashMap;

fn main() {
    let mut functions = HashMap::new();
    functions.insert("relu", FuncMeta::new(relu, "Rectified Linear Unit"));
    functions.insert("sigmoid", FuncMeta::new(sigmoid, "Logistic sigmoid"));
    let function_registry = FunctionRegistry::new(functions);

    let stages: Vec<Stage> = vec![
        Stage::new(scale(0.75)),
        Stage::new(bias(0.10)),
        Stage::from_arc(
            function_registry
                .get("relu")
                .expect("missing registry function: relu"),
        ),
    ];
    let output: Stage = Stage::from_arc(
        function_registry
            .get("sigmoid")
            .expect("missing registry function: sigmoid"),
    );

    let neuron = Neuron::new(stages, output);

    let x = -0.20f32;
    let y = neuron.forward(x);
    println!("forward({x}) = {y}");
}
