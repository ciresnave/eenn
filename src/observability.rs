use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub trait Profiler: Send + Sync {
    fn record_event(&self, name: &str, micros: u64);
    fn snapshot(&self) -> ProfilerSnapshot;
}

#[derive(Clone, Debug, Default)]
pub struct ProfilerSnapshot {
    pub events: HashMap<String, u64>,
}

#[derive(Clone)]
pub struct InMemoryProfiler(Arc<Mutex<HashMap<String, u64>>>);

impl InMemoryProfiler {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(HashMap::new())))
    }
}

impl Profiler for InMemoryProfiler {
    fn record_event(&self, name: &str, micros: u64) {
        let mut m = self.0.lock().unwrap();
        *m.entry(name.to_string()).or_insert(0) += micros;
    }
    fn snapshot(&self) -> ProfilerSnapshot {
        let m = self.0.lock().unwrap();
        ProfilerSnapshot { events: m.clone() }
    }
}

use std::sync::Arc as StdArc;

pub struct CacheStats {
    pub hit: u64,
    pub miss: u64,
    pub evicted: u64,
}

pub struct ExecutionContext {
    pub profiler: Option<StdArc<dyn Profiler>>,
    pub debug_mode: bool,
    pub artifact_provenance: HashMap<String, String>,
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            profiler: None,
            debug_mode: false,
            artifact_provenance: HashMap::new(),
        }
    }
    pub fn with_profiler(p: StdArc<dyn Profiler>) -> Self {
        Self {
            profiler: Some(p),
            debug_mode: false,
            artifact_provenance: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn profiler_records_and_snapshots() {
        let p = Arc::new(InMemoryProfiler::new());
        p.record_event("compile_ms", 10);
        p.record_event("compile_ms", 5);
        let snap = p.snapshot();
        assert_eq!(snap.events.get("compile_ms").copied().unwrap_or(0), 15);
    }
}
