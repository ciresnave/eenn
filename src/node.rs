use crate::tensor::validate_port_tensors;
use crate::tensor::*;
use ndarray::{ArrayD, Axis, IxDyn, concatenate};
use rand::rngs::StdRng;
use std::sync::Arc;

pub struct TickCtx<'a> {
    pub rng: &'a mut StdRng,
}

pub struct NodeSpec {
    pub inputs: Vec<PortSpec>,
    pub outputs: Vec<PortSpec>,
}

pub trait Node: Send {
    /// Return owned spec (inputs, outputs) for this node.
    fn spec(&self) -> NodeSpec;
    fn forward(&mut self, ctx: &mut TickCtx, inputs: PortTensors) -> anyhow::Result<PortTensors>;
}

pub struct Sequential {
    pub stages: Vec<Box<dyn Node>>,
}

impl Sequential {
    pub fn new(stages: Vec<Box<dyn Node>>) -> Self {
        Self { stages }
    }
}

impl Node for Sequential {
    fn spec(&self) -> NodeSpec {
        // Compose specs is non-trivial; return an empty/placeholder spec.
        NodeSpec {
            inputs: vec![],
            outputs: vec![],
        }
    }

    fn forward(&mut self, ctx: &mut TickCtx, inputs: PortTensors) -> anyhow::Result<PortTensors> {
        let mut buf = inputs;
        for stage in self.stages.iter_mut() {
            // Validate against stage input specs before running.
            let spec = stage.spec();
            if !spec.inputs.is_empty() {
                // If the buffer doesn't contain the expected input keys but has exactly
                // one entry, map that single entry to the expected single input id.
                let mut view = buf.clone();
                if spec.inputs.len() == 1
                    && !view.contains_key(&spec.inputs[0].id)
                    && view.len() == 1
                {
                    // take the sole entry and clone it under the expected id
                    let (_, v) = view.iter().next().unwrap();
                    view.insert(spec.inputs[0].id.clone(), v.clone());
                    // allow validation against `view`
                    validate_port_tensors(&spec.inputs, &view).map_err(|e| anyhow::anyhow!(e))?;
                } else {
                    validate_port_tensors(&spec.inputs, &view).map_err(|e| anyhow::anyhow!(e))?;
                }
            }
            buf = stage.forward(ctx, buf)?;
        }
        Ok(buf)
    }
}

// ReshapeNode: takes a single input tensor on port `in` and emits a single
// output tensor on port `out` with the requested shape. Only supports
// f32-backed tensors and only when the number of elements matches.
pub struct ReshapeNode {
    pub name: &'static str,
    pub in_port: PortId,
    pub out_port: PortId,
    pub target: Shape,
}

impl ReshapeNode {
    pub fn new(name: &'static str, in_port: PortId, out_port: PortId, target: Shape) -> Self {
        Self {
            name,
            in_port,
            out_port,
            target,
        }
    }
}

impl Node for ReshapeNode {
    fn spec(&self) -> NodeSpec {
        let input = PortSpec {
            id: self.in_port.clone(),
            arity: Arity::Exactly(1),
            dtype: Some(DType::F32),
            shape: None,
            allow_broadcast: false,
        };
        let output = PortSpec {
            id: self.out_port.clone(),
            arity: Arity::Exactly(1),
            dtype: Some(DType::F32),
            shape: Some(self.target.clone()),
            allow_broadcast: false,
        };
        NodeSpec {
            inputs: vec![input],
            outputs: vec![output],
        }
    }

    fn forward(
        &mut self,
        _ctx: &mut TickCtx,
        mut inputs: PortTensors,
    ) -> anyhow::Result<PortTensors> {
        let mut out = PortTensors::new();
        let v = inputs
            .remove(&self.in_port)
            .ok_or_else(|| anyhow::anyhow!("missing input"))?;
        if v.len() != 1 {
            return Err(anyhow::anyhow!("reshape expects exactly one tensor"));
        }
        let t = v.into_iter().next().unwrap();
        let src_nelems = t
            .shape
            .num_elements()
            .ok_or_else(|| anyhow::anyhow!("source shape unknown"))?;
        let tgt_nelems = self
            .target
            .num_elements()
            .ok_or_else(|| anyhow::anyhow!("target shape unknown"))?;
        if src_nelems != tgt_nelems {
            return Err(anyhow::anyhow!("element count mismatch for reshape"));
        }
        // Move data through by reusing storage where possible. For ndarray-backed
        // storage we attempt to reshape via ndarray; for device-backed storage we
        // try to materialize to host if possible.
        let new_storage = match &t.storage {
            TensorStorage::NdF32(a) => {
                // attempt to reshape ndarray
                let dims: Option<Vec<usize>> = self.target.0.iter().copied().collect();
                if let Some(dims) = dims {
                    let reshaped = a
                        .as_ref()
                        .clone()
                        .into_shape(IxDyn(&dims))
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    TensorStorage::NdF32(Arc::new(reshaped))
                } else {
                    // can't reshape with unknown dims: fall back to clone
                    TensorStorage::NdF32(a.clone())
                }
            }
            TensorStorage::Device(dev) => {
                // Attempt to pull device tensor to host and reshape
                if dev.dtype() == DType::F32 {
                    if let Ok(data) = dev.to_host_f32() {
                        if let Some(dims) = self
                            .target
                            .0
                            .iter()
                            .copied()
                            .collect::<Option<Vec<usize>>>()
                        {
                            let arr = ArrayD::from_shape_vec(IxDyn(&dims), data)
                                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                            TensorStorage::NdF32(Arc::new(arr))
                        } else {
                            // unknown dims: materialize as-is (can't reshape)
                            let arr = ArrayD::from_shape_vec(IxDyn(&[]), data)
                                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                            TensorStorage::NdF32(Arc::new(arr))
                        }
                    } else {
                        return Err(anyhow::anyhow!("reshape: unable to fetch device data"));
                    }
                } else {
                    return Err(anyhow::anyhow!("reshape: unsupported device dtype"));
                }
            }
            TensorStorage::NdF64(a) => {
                // shape dtype mismatch; cannot produce f32 ndarray here — keep as-is
                TensorStorage::NdF64(a.clone())
            }
        };
        let newt = Tensor {
            dtype: t.dtype,
            shape: self.target.clone(),
            storage: new_storage,
        };
        out.insert(self.out_port.clone(), vec![newt]);
        Ok(out)
    }
}

// ConcatNode: concatenates vectors on multiple ports into a single output port.
pub struct ConcatNode {
    pub name: &'static str,
    pub in_ports: Vec<PortId>,
    pub out_port: PortId,
    pub axis: usize, // only axis 0 supported for 1-D
}

impl ConcatNode {
    pub fn new(name: &'static str, in_ports: Vec<PortId>, out_port: PortId) -> Self {
        Self {
            name,
            in_ports,
            out_port,
            axis: 0,
        }
    }
}

impl Node for ConcatNode {
    fn spec(&self) -> NodeSpec {
        let inputs = self
            .in_ports
            .iter()
            .map(|p| PortSpec {
                id: p.clone(),
                arity: Arity::Exactly(1),
                dtype: Some(DType::F32),
                shape: None,
                allow_broadcast: false,
            })
            .collect();
        let output = PortSpec {
            id: self.out_port.clone(),
            arity: Arity::Exactly(1),
            dtype: Some(DType::F32),
            shape: None,
            allow_broadcast: false,
        };
        NodeSpec {
            inputs,
            outputs: vec![output],
        }
    }

    fn forward(
        &mut self,
        _ctx: &mut TickCtx,
        mut inputs: PortTensors,
    ) -> anyhow::Result<PortTensors> {
        // Use ndarray concat: collect ArrayD<f32> for each input tensor (first element)
        let mut arrays: Vec<ndarray::ArrayD<f32>> = Vec::new();
        for p in &self.in_ports {
            if let Some(mut vs) = inputs.remove(p) {
                for t in vs.drain(..) {
                    if let Some(a) = t.to_ndarray_f32() {
                        arrays.push((*a).clone());
                    } else {
                        // fallback: try to build ndarray from shape and any slice data
                        if let Some(slice) = t.as_f32_slice() {
                            if let Some(dims) =
                                t.shape.0.iter().copied().collect::<Option<Vec<usize>>>()
                            {
                                let arr = ArrayD::from_shape_vec(IxDyn(&dims), slice.to_vec())
                                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                                arrays.push(arr);
                            } else {
                                return Err(anyhow::anyhow!("concat: tensor has unknown shape"));
                            }
                        } else {
                            return Err(anyhow::anyhow!("concat expects f32-compatible tensors"));
                        }
                    }
                }
            }
        }
        if arrays.is_empty() {
            return Err(anyhow::anyhow!("concat: no inputs"));
        }
        // perform concatenation along axis
        let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
        let cat =
            concatenate(Axis(self.axis), &views).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let outt = Tensor {
            dtype: DType::F32,
            shape: Shape(cat.shape().iter().map(|d| Some(*d)).collect()),
            storage: TensorStorage::NdF32(Arc::new(cat)),
        };
        let mut out = PortTensors::new();
        out.insert(self.out_port.clone(), vec![outt]);
        Ok(out)
    }
}

// MergeNode: merges multiple tensors from a single port into multiple outputs
// under different keys. Simple mapping: input port -> copies to out_ports in order.
pub struct MergeNode {
    pub name: &'static str,
    pub in_port: PortId,
    pub out_ports: Vec<PortId>,
}

impl MergeNode {
    pub fn new(name: &'static str, in_port: PortId, out_ports: Vec<PortId>) -> Self {
        Self {
            name,
            in_port,
            out_ports,
        }
    }
}

impl Node for MergeNode {
    fn spec(&self) -> NodeSpec {
        let input = PortSpec {
            id: self.in_port.clone(),
            arity: Arity::Range { min: 1, max: None },
            dtype: None,
            shape: None,
            allow_broadcast: false,
        };
        let outputs = self
            .out_ports
            .iter()
            .map(|p| PortSpec {
                id: p.clone(),
                arity: Arity::Exactly(1),
                dtype: None,
                shape: None,
                allow_broadcast: false,
            })
            .collect();
        NodeSpec {
            inputs: vec![input],
            outputs,
        }
    }

    fn forward(
        &mut self,
        _ctx: &mut TickCtx,
        mut inputs: PortTensors,
    ) -> anyhow::Result<PortTensors> {
        let mut out = PortTensors::new();
        let v = inputs
            .remove(&self.in_port)
            .ok_or_else(|| anyhow::anyhow!("missing input"))?;
        // distribute in order (clone tensors by Arc)
        for (i, outp) in self.out_ports.iter().enumerate() {
            let t = v
                .get(i)
                .cloned()
                .or_else(|| v.first().cloned())
                .ok_or_else(|| anyhow::anyhow!("not enough tensors to merge"))?;
            out.insert(outp.clone(), vec![t]);
        }
        Ok(out)
    }
}

// Adapter to wrap existing Stage (Arc<dyn Fn(f32)->f32>) into the Node system.
pub struct StageNode {
    pub name: &'static str,
    pub stage: crate::Stage,
    spec: crate::tensor::PortSpec,
}

impl StageNode {
    pub fn new(name: &'static str, stage: crate::Stage) -> Self {
        let spec = crate::tensor::PortSpec {
            id: crate::tensor::PortId("x"),
            arity: crate::tensor::Arity::Exactly(1),
            dtype: Some(crate::tensor::DType::F32),
            shape: None,
            allow_broadcast: false,
        };
        Self { name, stage, spec }
    }
}

impl Node for StageNode {
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            inputs: vec![self.spec.clone()],
            outputs: vec![crate::tensor::PortSpec {
                id: PortId("y"),
                arity: Arity::Exactly(1),
                dtype: Some(DType::F32),
                shape: None,
                allow_broadcast: false,
            }],
        }
    }

    fn forward(
        &mut self,
        _ctx: &mut TickCtx,
        mut inputs: PortTensors,
    ) -> anyhow::Result<PortTensors> {
        let mut out = PortTensors::new();
        // Accept input on either the standard input port "x" or the
        // previous-stage output port "y" so Sequential pipelines can chain
        // StageNode instances without additional wiring.
        let maybe_vec = inputs
            .remove(&crate::tensor::PortId("x"))
            .or_else(|| inputs.remove(&crate::tensor::PortId("y")));

        if let Some(mut v) = maybe_vec {
            let t = v.pop().unwrap();
            // Only support scalar f32 for now
            if let Some(x) = t.as_f32_scalar() {
                let yv = self.stage.call(x);
                out.insert(
                    crate::tensor::PortId("y"),
                    vec![crate::tensor::Tensor::scalar_f32(yv)],
                );
                return Ok(out);
            }
        }
        Err(anyhow::anyhow!("StageNode expected scalar f32 input"))
    }
}
