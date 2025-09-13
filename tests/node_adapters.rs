use eenn::tensor::{Arity, validate_port_tensors};
use eenn::*;
use rand::SeedableRng;

#[test]
fn reshape_concat_merge_happy_path() {
    // prepare a vector tensor on port "a"
    let mut inputs = PortTensors::new();
    inputs.insert(PortId("a"), vec![Tensor::vector_f32(vec![1.0, 2.0, 3.0])]);

    // Reshape 3 -> (3,) (no op)
    let mut r = ReshapeNode::new("r", PortId("a"), PortId("b"), Shape::vector(3));
    let mut ctx = TickCtx {
        rng: &mut rand::rngs::StdRng::seed_from_u64(0),
    };
    let out = r.forward(&mut ctx, inputs).expect("reshape");
    let t = out.get(&PortId("b")).unwrap().first().unwrap();
    assert!(t.is_vector());
    assert_eq!(t.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0]);

    // Concat two ports
    let mut inputs2 = PortTensors::new();
    inputs2.insert(PortId("p1"), vec![Tensor::vector_f32(vec![1.0, 2.0])]);
    inputs2.insert(PortId("p2"), vec![Tensor::vector_f32(vec![3.0, 4.0])]);
    let mut c = ConcatNode::new("c", vec![PortId("p1"), PortId("p2")], PortId("out"));
    let out2 = c.forward(&mut ctx, inputs2).expect("concat");
    let t2 = out2.get(&PortId("out")).unwrap().first().unwrap();
    assert_eq!(t2.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    // Merge: single input with two outputs
    let mut inputs3 = PortTensors::new();
    inputs3.insert(
        PortId("in"),
        vec![Tensor::scalar_f32(7.0), Tensor::scalar_f32(8.0)],
    );
    let mut m = MergeNode::new("m", PortId("in"), vec![PortId("o1"), PortId("o2")]);
    let out3 = m.forward(&mut ctx, inputs3).expect("merge");
    let a = out3
        .get(&PortId("o1"))
        .unwrap()
        .first()
        .unwrap()
        .as_f32_scalar()
        .unwrap();
    let b = out3
        .get(&PortId("o2"))
        .unwrap()
        .first()
        .unwrap()
        .as_f32_scalar()
        .unwrap();
    assert_eq!(a, 7.0);
    assert_eq!(b, 8.0);
}

#[test]
fn validate_port_tensors_errors() {
    // arity mismatch
    let spec = PortSpec {
        id: PortId("x"),
        arity: Arity::Exactly(2),
        dtype: Some(DType::F32),
        shape: None,
        allow_broadcast: false,
    };
    let mut tensors = PortTensors::new();
    tensors.insert(PortId("x"), vec![Tensor::scalar_f32(1.0)]);
    let res = validate_port_tensors(&[spec], &tensors);
    assert!(res.is_err());
}
