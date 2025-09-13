use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Calibration {
    pub bandwidth_bytes_per_sec: f64,
}

impl Calibration {
    pub fn bytes_per_ms(&self) -> f64 {
        self.bandwidth_bytes_per_sec / 1000.0
    }

    pub fn load_from_cache_root(root: impl AsRef<Path>) -> anyhow::Result<Option<Self>> {
        let p = root.as_ref().join("calibration.json");
        if !p.exists() {
            return Ok(None);
        }
        let s = fs::read_to_string(&p).context("read calibration file")?;
        let c: Calibration = serde_json::from_str(&s).context("parse calibration json")?;
        Ok(Some(c))
    }

    pub fn persist_to_cache_root(&self, root: impl AsRef<Path>) -> anyhow::Result<()> {
        let p = root.as_ref().join("calibration.json");
        let s = serde_json::to_string_pretty(self)?;
        fs::write(&p, s).context("write calibration file")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn persist_and_load_calibration() {
        let dir = tempdir().unwrap();
        let c = Calibration {
            bandwidth_bytes_per_sec: 100.0,
        };
        c.persist_to_cache_root(dir.path()).unwrap();
        let loaded = Calibration::load_from_cache_root(dir.path())
            .unwrap()
            .unwrap();
        assert_eq!(loaded.bandwidth_bytes_per_sec, 100.0);
    }
}
