use eenn::{Node, PortId, PortTensors, Sequential, StageNode, Tensor, TickCtx};
use eenn::{Stage, relu};
use rand::SeedableRng;

#[test]
fn sequential_stage_pipeline_matches_neuron() {
    // Build a Sequential of two StageNodes: relu then scale(0.5)
    let relu_stage = Stage::new(relu);
    let scale_stage = Stage::new(|x| 0.5 * x);

    let nodes: Vec<Box<dyn eenn::node::Node>> = vec![
        Box::new(StageNode::new("relu", relu_stage)),
        Box::new(StageNode::new("scale", scale_stage)),
    ];

    let mut seq = Sequential::new(nodes);

    // prepare input
    let mut input_map = PortTensors::new();
    input_map.insert(PortId("x"), vec![Tensor::scalar_f32(2.0)]);

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut ctx = TickCtx { rng: &mut rng };

    let out = seq.forward(&mut ctx, input_map).expect("forward");
    let v = out
        .get(&PortId("y"))
        .unwrap()
        .first()
        .unwrap()
        .as_f32_scalar()
        .unwrap();
    assert_eq!(v, 1.0);
}
