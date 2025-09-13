use anyhow::Context;
use base64::{Engine as _, engine::general_purpose};
use ed25519_dalek::{PublicKey, Signature, Verifier};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Minimal trait representing a local trust store of public keys.
pub trait TrustStore: Send + Sync {
    /// Return the public key bytes for a given signer id if present.
    fn get_public_key(&self, signer: &str) -> Option<Vec<u8>>;
}

/// In-memory trust store for tests and simple deployments.
#[derive(Default, Clone)]
pub struct InMemoryTrustStore {
    inner: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

impl InMemoryTrustStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn insert(&self, signer: impl Into<String>, pk: Vec<u8>) {
        let mut g = self.inner.lock().unwrap();
        g.insert(signer.into(), pk);
    }
}

impl TrustStore for InMemoryTrustStore {
    fn get_public_key(&self, signer: &str) -> Option<Vec<u8>> {
        let g = self.inner.lock().unwrap();
        g.get(signer).cloned()
    }
}

/// Manifest wrapper expected to include signer metadata and base64 signature.
#[derive(Deserialize)]
struct SignedManifest {
    signer: String,
    signature: String, // base64 of ed25519 signature
                       // other fields ignored for verification; the manifest bytes to verify should
                       // be provided separately as canonical JSON bytes.
}

/// Verify a canonical manifest bytes against an expected SignedManifest JSON
/// which contains `signer` and `signature` fields. The `signed_manifest_json`
/// is a JSON string that at minimum contains those fields (it may also include
/// other metadata). The `canonical_manifest_bytes` are the canonical JSON bytes
/// that were signed.
pub fn verify_manifest(
    signed_manifest_json: &str,
    canonical_manifest_bytes: &[u8],
    trust: &dyn TrustStore,
) -> anyhow::Result<()> {
    let sm: SignedManifest =
        serde_json::from_str(signed_manifest_json).context("parse signed manifest")?;
    let sig_bytes = general_purpose::STANDARD
        .decode(&sm.signature)
        .context("decode signature")?;
    let sig = Signature::from_bytes(&sig_bytes).context("parse signature")?;
    let pk_bytes = trust
        .get_public_key(&sm.signer)
        .ok_or_else(|| anyhow::anyhow!("unknown signer"))?;
    let pk = PublicKey::from_bytes(&pk_bytes).context("parse public key")?;
    pk.verify(canonical_manifest_bytes, &sig)
        .context("signature verification failed")?;
    Ok(())
}

/// Verify the base64-ed25519 signature on the provided canonical manifest bytes.
/// `signature_b64` is expected to be the base64 encoding of the 64-byte ed25519 signature.
pub fn verify_manifest_signature(
    manifest_bytes: &[u8],
    signature_b64: &str,
    public_key_bytes: &[u8],
) -> anyhow::Result<()> {
    let sig_bytes = general_purpose::STANDARD
        .decode(signature_b64)
        .context("decoding base64 signature")?;
    let signature = Signature::from_bytes(&sig_bytes).context("parsing signature bytes")?;
    let pubkey = PublicKey::from_bytes(public_key_bytes).context("parsing public key bytes")?;
    pubkey
        .verify(manifest_bytes, &signature)
        .context("signature verification failed")?;
    Ok(())
}

/// Helper: extract canonical bytes for signing from a serde_json::Value by re-serializing
/// with stable key order. This mirrors `to_canonical_json` in `fingerprint.rs` but here
/// kept small to avoid extra public API coupling.
pub fn canonical_bytes_from_value(v: &Value) -> anyhow::Result<Vec<u8>> {
    // Serialize using serde_json::to_value + sort keys as in fingerprint module
    let mut value = v.clone();
    sort_json_value_keys(&mut value);
    let s = serde_json::to_vec(&value)?;
    Ok(s)
}

fn sort_json_value_keys(v: &mut Value) {
    match v {
        Value::Object(map) => {
            // Collect entries (cloning values) then clear and reinsert in sorted order.
            let mut entries: Vec<_> = map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            entries.sort_by_key(|(k, _)| k.clone());
            map.clear();
            for (k, mut val) in entries {
                sort_json_value_keys(&mut val);
                map.insert(k, val);
            }
        }
        Value::Array(a) => {
            for item in a.iter_mut() {
                sort_json_value_keys(item);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Keypair;
    use ed25519_dalek::PublicKey as DalekPublicKey;
    use ed25519_dalek::SecretKey;
    use ed25519_dalek::Signer as _;

    #[test]
    fn verify_signed_manifest_happy_path() {
        // deterministic keypair derived from a fixed 32-byte seed
        let seed = [0x12u8; 32];
        let sk = SecretKey::from_bytes(&seed).expect("secret key");
        let pubkey = DalekPublicKey::from(&sk);
        let kp = Keypair {
            secret: sk,
            public: pubkey,
        };

        // canonical manifest to sign
        let manifest = b"{\"ops\":[],\"version\":1}";

        // sign
        let sig = kp.sign(manifest);

        let signed_json = serde_json::json!({
            "signer": "test-signer",
            "signature": general_purpose::STANDARD.encode(sig.to_bytes()),
        })
        .to_string();

        let trust = InMemoryTrustStore::new();
        trust.insert("test-signer", kp.public.to_bytes().to_vec());

        verify_manifest(&signed_json, manifest, &trust as &dyn TrustStore).unwrap();
    }

    #[test]
    fn verify_rejects_tampered_manifest() {
        let seed = [0x34u8; 32];
        let sk = SecretKey::from_bytes(&seed).expect("secret key");
        let pubkey = DalekPublicKey::from(&sk);
        let kp = Keypair {
            secret: sk,
            public: pubkey,
        };
        let manifest = b"{\"ops\":[],\"version\":1}";
        let sig = kp.sign(manifest);
        let signed_json = serde_json::json!({
            "signer": "test-signer",
            "signature": general_purpose::STANDARD.encode(sig.to_bytes()),
        })
        .to_string();

        let trust = InMemoryTrustStore::new();
        trust.insert("test-signer", kp.public.to_bytes().to_vec());

        // tamper manifest bytes
        let tampered = b"{\"ops\":[],\"version\":2}";
        let res = verify_manifest(&signed_json, tampered, &trust as &dyn TrustStore);
        assert!(res.is_err());
    }

    #[test]
    fn sign_and_verify_manifest_roundtrip() {
        // Create a deterministic keypair for the test
        let seed = [0x12u8; 32];
        let sk = SecretKey::from_bytes(&seed).expect("secret key");
        let pubkey = DalekPublicKey::from(&sk);
        let kp = Keypair {
            secret: sk,
            public: pubkey,
        };

        // Create a sample manifest as JSON Value
        let manifest = serde_json::json!({
            "fingerprint_version": 1,
            "ops": [ { "name": "relu", "impl": "v1" } ],
            "dtype": "f32",
            "composer_version": "wgsl-composer-0.1",
            "artifact_kind": "generic",
        });

        let bytes = canonical_bytes_from_value(&manifest).expect("canonicalize");

        // Sign
        let sig = kp.sign(&bytes);
        let sig_b64 = general_purpose::STANDARD.encode(sig.to_bytes());

        // Verify using the helper
        let pubkey_bytes = kp.public.to_bytes();
        verify_manifest_signature(&bytes, &sig_b64, &pubkey_bytes).expect("verify ok");

        // Also check TrustStore lookup path
        let signed_json = serde_json::json!({
            "signer": "test-signer",
            "signature": sig_b64,
        })
        .to_string();

        let trust = InMemoryTrustStore::new();
        trust.insert("test-signer", pubkey_bytes.to_vec());
        verify_manifest(&signed_json, &bytes, &trust as &dyn TrustStore)
            .expect("verify via truststore");
    }
}
