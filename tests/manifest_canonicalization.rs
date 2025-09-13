use eenn::{fingerprint_sha256_hex, to_canonical_json};
use serde_json::json;

#[test]
fn permuted_keys_produce_same_fingerprint() {
    let a = json!({"ops": ["relu", "scale"], "version": 1, "device_features": ["f32"]});
    let b = json!({"device_features": ["f32"], "ops": ["relu", "scale"], "version": 1});
    let fa = fingerprint_sha256_hex(&a).unwrap();
    let fb = fingerprint_sha256_hex(&b).unwrap();
    assert_eq!(fa, fb, "fingerprints must match for permuted keys");
}

#[test]
fn differently_formatted_floats_produce_same_fingerprint() {
    // Create two JSON strings with different float formatting but same numeric value
    let s1 = r#"{"value": 1.0, "ratio": 0.10000000000000001}"#;
    let s2 = r#"{"ratio": 0.1, "value": 1}"#;
    let v1: serde_json::Value = serde_json::from_str(s1).expect("parse s1");
    let v2: serde_json::Value = serde_json::from_str(s2).expect("parse s2");
    // canonical JSON strings should be equal and fingerprints equal
    let c1 = to_canonical_json(&v1).unwrap();
    let c2 = to_canonical_json(&v2).unwrap();
    assert_eq!(c1, c2, "canonical JSON should normalize float formatting");
    let f1 = fingerprint_sha256_hex(&v1).unwrap();
    let f2 = fingerprint_sha256_hex(&v2).unwrap();
    assert_eq!(
        f1, f2,
        "fingerprints should match for numerically equal floats"
    );
}

#[test]
fn explicit_null_vs_missing_field_are_distinct() {
    let a = json!({"ops": ["relu"], "optional": null});
    let b = json!({"ops": ["relu"]});
    // canonical JSON differs because `optional` is present in `a` but missing in `b`
    let ca = to_canonical_json(&a).unwrap();
    let cb = to_canonical_json(&b).unwrap();
    assert_ne!(ca, cb, "explicit null should differ from missing field");
    let fa = fingerprint_sha256_hex(&a).unwrap();
    let fb = fingerprint_sha256_hex(&b).unwrap();
    assert_ne!(
        fa, fb,
        "fingerprints should differ when optional field is null vs missing"
    );
}
