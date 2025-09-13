//! Models and serialization helpers.
//!
//! When the `rkyv` feature is enabled we use rkyv for zero-copy (de)serialization.
//! Otherwise we fall back to serde + bincode. The public API is identical.

#[cfg(feature = "rkyv")]
mod imp {
    // prefer the convenience helper when available
    use rkyv::rancor::Error as RkyvError;
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Debug, PartialEq, Clone, Archive, Serialize, Deserialize)]
    pub enum StageDef {
        Named(String),
        Scale { w: f32 },
        Bias { b: f32 },
        Factory { name: String, params: Vec<f32> },
    }

    #[derive(Debug, PartialEq, Clone, Archive, Serialize, Deserialize)]
    pub struct NeuronDef {
        pub version: u32,
        pub stages: Vec<StageDef>,
        pub output: StageDef,
    }

    impl NeuronDef {
        pub fn new(stages: Vec<StageDef>, output: StageDef) -> Self {
            Self {
                version: 1,
                stages,
                output,
            }
        }

        pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
            rkyv::to_bytes(self)
                .map_err(|e: RkyvError| format!("rkyv serialize error: {}", e))
                .map(|v| v.into_vec())
        }

        pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
            unsafe {
                rkyv::from_bytes_unchecked::<Self, rkyv::rancor::Error>(bytes)
                    .map_err(|e: rkyv::rancor::Error| format!("rkyv deserialize error: {}", e))
            }
        }

        pub fn to_neuron(
            &self,
            registry: &crate::FunctionRegistry,
        ) -> Result<crate::Neuron, String> {
            use crate::{Stage, bias, scale};

            let mut stages = Vec::new();
            for s in &self.stages {
                match s {
                    StageDef::Named(name) => {
                        if let Some(f) = registry.get(name) {
                            stages.push(Stage::from_arc(f));
                        } else {
                            return Err(format!("unknown function '{}'", name));
                        }
                    }
                    StageDef::Scale { w } => stages.push(Stage::new(scale(*w))),
                    StageDef::Bias { b } => stages.push(Stage::new(bias(*b))),
                    StageDef::Factory { name, params } => {
                        if let Some(f) = registry.call_factory(name, params) {
                            stages.push(Stage::from_arc(f));
                        } else {
                            return Err(format!("unknown factory '{}'", name));
                        }
                    }
                }
            }

            let output_stage = match &self.output {
                StageDef::Named(name) => {
                    if let Some(f) = registry.get(name) {
                        Stage::from_arc(f)
                    } else {
                        return Err(format!("unknown function '{}'", name));
                    }
                }
                StageDef::Scale { w } => Stage::new(scale(*w)),
                StageDef::Bias { b } => Stage::new(bias(*b)),
                StageDef::Factory { name, params } => {
                    if let Some(f) = registry.call_factory(name, params) {
                        Stage::from_arc(f)
                    } else {
                        return Err(format!("unknown factory '{}'", name));
                    }
                }
            };

            Ok(crate::Neuron::new(stages, output_stage))
        }

        /// Zero-copy-ish rehydration: validate the buffer and walk the archived
        /// view to construct a `Neuron` without allocating owned `NeuronDef` or
        /// `StageDef` instances. Primitive numeric values are copied where
        /// necessary (small cost); strings and vec headers remain borrowed.
        pub fn from_bytes_zero_copy_to_neuron(
            bytes: &[u8],
            registry: &crate::FunctionRegistry,
        ) -> Result<crate::Neuron, String> {
            use crate::{Stage, bias, scale};
            use rkyv::api::high::access;

            // Validate and get a borrowed archived view. `access` returns
            // a Result<&ArchivedT, Error> when the `bytecheck` feature is
            // enabled (we enabled it in Cargo.toml for the git dependency).
            let archived = access::<ArchivedNeuronDef, rkyv::rancor::Error>(bytes)
                .map_err(|e| format!("rkyv access error: {}", e))?;

            let mut stages = Vec::new();

            // archived.stages is an Archived<Vec<ArchivedStageDef>> which
            // exposes as_slice for borrowing.
            for s in archived.stages.as_slice().iter() {
                match s {
                    ArchivedStageDef::Named(name) => {
                        let name_str: &str = name.as_str();
                        if let Some(f) = registry.get(name_str) {
                            stages.push(Stage::from_arc(f));
                        } else {
                            return Err(format!("unknown function '{}'", name_str));
                        }
                    }
                    ArchivedStageDef::Scale { w } => {
                        // archived numeric wrappers convert into native types
                        // via Into::<f32>::into or to_native depending on
                        // the underlying wrapper. `into()` works for the
                        // rend endian wrappers.
                        let w_val: f32 = (*w).into();
                        stages.push(Stage::new(scale(w_val)));
                    }
                    ArchivedStageDef::Bias { b } => {
                        let b_val: f32 = (*b).into();
                        stages.push(Stage::new(bias(b_val)));
                    }
                    ArchivedStageDef::Factory { name, params } => {
                        let name_str: &str = name.as_str();
                        // map archived param slice to owned Vec<f32>
                        let params_vec: Vec<f32> =
                            params.as_slice().iter().map(|v| (*v).into()).collect();
                        if let Some(f) = registry.call_factory(name_str, &params_vec) {
                            stages.push(Stage::from_arc(f));
                        } else {
                            return Err(format!("unknown factory '{}'", name_str));
                        }
                    }
                }
            }

            let output_stage = match &archived.output {
                ArchivedStageDef::Named(name) => {
                    let name_str: &str = name.as_str();
                    if let Some(f) = registry.get(name_str) {
                        Stage::from_arc(f)
                    } else {
                        return Err(format!("unknown function '{}'", name_str));
                    }
                }
                ArchivedStageDef::Scale { w } => {
                    let w_val: f32 = (*w).into();
                    Stage::new(scale(w_val))
                }
                ArchivedStageDef::Bias { b } => {
                    let b_val: f32 = (*b).into();
                    Stage::new(bias(b_val))
                }
                ArchivedStageDef::Factory { name, params } => {
                    let name_str: &str = name.as_str();
                    let params_vec: Vec<f32> =
                        params.as_slice().iter().map(|v| (*v).into()).collect();
                    if let Some(f) = registry.call_factory(name_str, &params_vec) {
                        Stage::from_arc(f)
                    } else {
                        return Err(format!("unknown factory '{}'", name_str));
                    }
                }
            };

            Ok(crate::Neuron::new(stages, output_stage))
        }
    }

    // Unsafe unchecked rehydrate path: deserialize without validation and
    // convert to a runtime `Neuron`. This path is behind the
    // `rkyv_unchecked` feature and should only be enabled for trusted data.
    #[cfg(feature = "rkyv_unchecked")]
    impl NeuronDef {
        /// Unsafe unchecked rehydrate: deserialize without validation and
        /// convert to a runtime `Neuron`. This avoids the bytecheck
        /// validation overhead by using `from_bytes_unchecked`. It returns
        /// owned `NeuronDef` instances (no zero-copy), but is faster for
        /// trusted inputs.
        /// # Safety
        ///
        /// Caller must ensure `bytes` contains a valid archived `NeuronDef`.
        /// This function uses `rkyv::from_bytes_unchecked` which does not
        /// perform validation; providing invalid data is undefined behavior.
        pub unsafe fn from_bytes_unchecked_to_neuron(
            bytes: &[u8],
            registry: &crate::FunctionRegistry,
        ) -> Result<crate::Neuron, String> {
            // SAFETY: caller is responsible for ensuring `bytes` is a valid
            // archived representation of `NeuronDef`.
            let def: Self = unsafe {
                rkyv::from_bytes_unchecked::<Self, rkyv::rancor::Error>(bytes).map_err(
                    |e: rkyv::rancor::Error| format!("rkyv unchecked deserialize error: {}", e),
                )?
            };
            def.to_neuron(registry)
        }
    }
}

#[cfg(not(feature = "rkyv"))]
mod imp {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
    pub enum StageDef {
        Named(String),
        Scale { w: f32 },
        Bias { b: f32 },
        Factory { name: String, params: Vec<f32> },
    }

    #[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
    pub struct NeuronDef {
        pub version: u32,
        pub stages: Vec<StageDef>,
        pub output: StageDef,
    }

    impl NeuronDef {
        pub fn new(stages: Vec<StageDef>, output: StageDef) -> Self {
            Self {
                version: 1,
                stages,
                output,
            }
        }

        pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
            bincode::serialize(self)
        }

        pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
            bincode::deserialize(bytes)
        }

        pub fn to_neuron(
            &self,
            registry: &crate::FunctionRegistry,
        ) -> Result<crate::Neuron, String> {
            use crate::{Stage, bias, scale};

            let mut stages = Vec::new();
            for s in &self.stages {
                match s {
                    StageDef::Named(name) => {
                        if let Some(f) = registry.get(name) {
                            stages.push(Stage::from_arc(f));
                        } else {
                            return Err(format!("unknown function '{}'", name));
                        }
                    }
                    StageDef::Scale { w } => stages.push(Stage::new(scale(*w))),
                    StageDef::Bias { b } => stages.push(Stage::new(bias(*b))),
                    StageDef::Factory { name, params } => {
                        if let Some(f) = registry.call_factory(name, params) {
                            stages.push(Stage::from_arc(f));
                        } else {
                            return Err(format!("unknown factory '{}'", name));
                        }
                    }
                }
            }

            let output_stage = match &self.output {
                StageDef::Named(name) => {
                    if let Some(f) = registry.get(name) {
                        Stage::from_arc(f)
                    } else {
                        return Err(format!("unknown function '{}'", name));
                    }
                }
                StageDef::Scale { w } => Stage::new(scale(*w)),
                StageDef::Bias { b } => Stage::new(bias(*b)),
                StageDef::Factory { name, params } => {
                    if let Some(f) = registry.call_factory(name, params) {
                        Stage::from_arc(f)
                    } else {
                        return Err(format!("unknown factory '{}'", name));
                    }
                }
            };

            Ok(crate::Neuron::new(stages, output_stage))
        }
    }
}

pub use imp::{NeuronDef, StageDef};
