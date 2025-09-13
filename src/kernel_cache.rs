use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ArtifactMeta {
    pub fingerprint: String,
    pub size_bytes: u64,
}

pub struct KernelCache {
    root: PathBuf,
    index: HashMap<String, ArtifactMeta>,
}

impl KernelCache {
    pub fn new(root: impl AsRef<Path>) -> anyhow::Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).context("create cache root")?;
        let mut kc = Self {
            root,
            index: HashMap::new(),
        };
        // Attempt to load existing index.json if present
        let _ = kc.load_index();
        Ok(kc)
    }

    pub fn artifact_path(&self, fingerprint: &str) -> PathBuf {
        self.root.join(format!("{}.bin", fingerprint))
    }

    /// Write bytes atomically to the cache and update in-memory index.
    pub fn write_artifact(&mut self, fingerprint: &str, bytes: &[u8]) -> anyhow::Result<()> {
        let target = self.artifact_path(fingerprint);
        let tmp = target.with_extension("tmp");
        fs::write(&tmp, bytes).context("write temp artifact")?;
        fs::rename(&tmp, &target).context("atomic rename")?;
        let meta = ArtifactMeta {
            fingerprint: fingerprint.to_string(),
            size_bytes: bytes.len() as u64,
        };
        self.index.insert(fingerprint.to_string(), meta);
        // Persist index atomically
        self.save_index()?;
        Ok(())
    }

    fn index_path(&self) -> PathBuf {
        self.root.join("index.json")
    }

    fn save_index(&self) -> anyhow::Result<()> {
        let p = self.index_path();
        let tmp = p.with_extension("tmp");
        let serialized = serde_json::to_vec(&self.index).context("serialize index")?;
        fs::write(&tmp, &serialized).context("write temp index")?;
        fs::rename(&tmp, &p).context("atomic rename index")?;
        Ok(())
    }

    fn load_index(&mut self) -> anyhow::Result<()> {
        let p = self.index_path();
        if p.exists() {
            let b = fs::read(&p).context("read index file")?;
            let map: HashMap<String, ArtifactMeta> =
                serde_json::from_slice(&b).context("parse index")?;
            self.index = map;
        }
        Ok(())
    }

    pub fn lookup(&self, fingerprint: &str) -> Option<&ArtifactMeta> {
        self.index.get(fingerprint)
    }

    pub fn read_artifact(&self, fingerprint: &str) -> anyhow::Result<Vec<u8>> {
        let p = self.artifact_path(fingerprint);
        let b = fs::read(&p).context("read artifact file")?;
        Ok(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn write_and_read_artifact_roundtrip() {
        let dir = tempdir().unwrap();
        let mut cache = KernelCache::new(dir.path()).unwrap();
        let fp = "deadbeef";
        let data = b"hello-world";
        cache.write_artifact(fp, data).unwrap();
        let meta = cache.lookup(fp).expect("meta present");
        assert_eq!(meta.size_bytes, data.len() as u64);
        let got = cache.read_artifact(fp).unwrap();
        assert_eq!(&got[..], &data[..]);
    }

    #[test]
    fn index_persisted_across_instances() {
        let dir = tempdir().unwrap();
        {
            let mut cache = KernelCache::new(dir.path()).unwrap();
            cache.write_artifact("one", b"a").unwrap();
            cache.write_artifact("two", b"bb").unwrap();
        }
        // new instance should load index.json
        let cache2 = KernelCache::new(dir.path()).unwrap();
        let m1 = cache2.lookup("one").expect("one present");
        let m2 = cache2.lookup("two").expect("two present");
        assert_eq!(m1.size_bytes, 1);
        assert_eq!(m2.size_bytes, 2);
    }
}
