use serde::Serialize;
use serde_json::Value;
use serde_json::ser::Serializer;
use sha2::{Digest, Sha256};

/// Produce a canonical JSON string with stable key ordering for objects.
pub fn to_canonical_json<T: Serialize>(v: &T) -> anyhow::Result<String> {
    // First serialize to a serde_json::Value so we can re-order object keys.
    let value = serde_json::to_value(v)?;
    let canonical = canonicalize_value(&value);
    let mut buf = Vec::with_capacity(256);
    let mut ser = Serializer::new(&mut buf);
    canonical.serialize(&mut ser)?;
    Ok(String::from_utf8(buf)?)
}

fn canonicalize_value(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            // Create a new map with sorted keys
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by_key(|(k, _)| (*k).clone());
            let mut new_map = serde_json::Map::with_capacity(entries.len());
            for (k, val) in entries {
                new_map.insert(k.clone(), canonicalize_value(val));
            }
            Value::Object(new_map)
        }
        Value::Array(a) => Value::Array(a.iter().map(canonicalize_value).collect()),
        Value::Number(n) => {
            // Normalize numbers by converting to f64-based representation so that
            // 1 and 1.0 serialize identically (as 1.0). This reduces accidental
            // fingerprint divergence due to formatting.
            if let Some(f) = n.as_f64() {
                // from_f64 returns None for NaN/inf — those should not occur in manifests
                if let Some(num) = serde_json::Number::from_f64(f) {
                    return Value::Number(num);
                }
            }
            // Fallback to clone if conversion fails
            v.clone()
        }
        _ => v.clone(),
    }
}

/// Compute SHA256 hex fingerprint of the canonical JSON serialization of `t`.
pub fn fingerprint_sha256_hex<T: Serialize>(t: &T) -> anyhow::Result<String> {
    let json = to_canonical_json(t)?;
    let mut hasher = Sha256::new();
    hasher.update(json.as_bytes());
    let digest = hasher.finalize();
    Ok(hex::encode(digest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Serialize)]
    struct Simple {
        b: u32,
        a: String,
    }

    #[test]
    fn canonical_json_stable_order() {
        let s = Simple {
            b: 2,
            a: "x".into(),
        };
        let j = to_canonical_json(&s).unwrap();
        // keys must be ordered alphabetically: a then b
        assert!(j.contains("\"a\""));
        assert!(j.find("\"a\"").unwrap() < j.find("\"b\"").unwrap());
    }

    #[test]
    fn fingerprint_is_deterministic() {
        let a = serde_json::json!({"z":1, "y":2});
        let b = serde_json::json!({"y":2, "z":1});
        let fa = fingerprint_sha256_hex(&a).unwrap();
        let fb = fingerprint_sha256_hex(&b).unwrap();
        assert_eq!(fa, fb);
    }
}
