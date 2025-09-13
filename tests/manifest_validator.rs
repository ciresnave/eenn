use anyhow::Context;
use std::fs;

#[test]
fn validate_sample_signed_manifest_against_schema() -> anyhow::Result<()> {
    // Load files from docs/
    let schema_text = fs::read_to_string("docs/manifest_schema.json")
        .context("reading docs/manifest_schema.json")?;
    let manifest_text = fs::read_to_string("docs/sample_signed_manifest.json")
        .context("reading docs/sample_signed_manifest.json")?;

    let schema_json: serde_json::Value =
        serde_json::from_str(&schema_text).context("parsing manifest_schema.json")?;
    let manifest_json: serde_json::Value =
        serde_json::from_str(&manifest_text).context("parsing sample_signed_manifest.json")?;

    // jsonschema::JSONSchema::compile borrows the schema. For a short-lived test
    // we can leak the parsed schema to obtain a 'static reference which the
    // compiled validator can hold. This is acceptable in tests.
    let static_schema: &'static serde_json::Value = Box::leak(Box::new(schema_json));
    let compiled =
        jsonschema::JSONSchema::compile(static_schema).context("compiling JSON Schema")?;

    let result = compiled.validate(&manifest_json);
    if let Err(errors) = result {
        for err in errors {
            eprintln!("schema validation error: {}", err);
        }
        anyhow::bail!("manifest did not validate against schema");
    }

    Ok(())
}
