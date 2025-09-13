eenn — tiny neural network building blocks

**NOT READY FOR PRODUCTION USE!**

Threading and safety

- All registered functions are stored as `Arc<dyn Fn(f32) -> f32 + Send + Sync + 'static>`.
- This means any function or closure you register must be `Send + Sync + 'static`.
- The `Arc` ensures clones are cheap and safe to share across threads.
- The `Fn` trait used (not `FnMut` or `FnOnce`) guarantees that the stored functions are callable concurrently.

Usage notes

- `FunctionRegistry::get(name)` returns an `Option<Arc<dyn Fn(...)>>`. Callers should handle the `None` case.
- If you prefer convenience, use `get(name).expect("message")` in simple scripts but avoid panics in library code.
- Register closures and plain functions with `register` and `register_fn`.

Example

```rust
use eenn::{FunctionRegistry, relu, scale};
let mut r = FunctionRegistry::empty();
r.register_fn("relu", relu, "ReLU");
r.register("scale_075", scale(0.75), "Scale by 0.75");
let f = r.get("scale_075").expect("found");
assert_eq!((f)(4.0f32), 3.0f32);
eenn — tiny neural network building blocks

Threading and safety

- All registered functions are stored as `Arc<dyn Fn(f32) -> f32 + Send + Sync + 'static>`.
- This means any function or closure you register must be `Send + Sync + 'static`.
- The `Arc` ensures clones are cheap and safe to share across threads.
- The `Fn` trait used (not `FnMut` or `FnOnce`) guarantees that the stored functions are callable concurrently.

Usage notes

- `FunctionRegistry::get(name)` returns an `Option<Arc<dyn Fn(...)>>`. Callers should handle the `None` case.
- If you prefer convenience, use `get(name).expect("message")` in simple scripts but avoid panics in library code.
- Register closures and plain functions with `register` and `register_fn`.

Example

```rust
use eenn::{FunctionRegistry, relu, scale};
let mut r = FunctionRegistry::empty();
r.register_fn("relu", relu, "ReLU");
r.register("scale_075", scale(0.75), "Scale by 0.75");
let f = r.get("scale_075").expect("found");
assert_eq!((f)(4.0f32), 3.0f32);
```

Pinned rkyv dependency
----------------------

This project uses rkyv for an optional zero-copy serialization path. The rkyv dependency is intentionally pinned to a specific git commit in `Cargo.toml` to ensure reproducible builds and because certain safe archived-access helpers used by the code require features that were not available in earlier published releases.

If you need to update rkyv, please do so deliberately: pick a released tag or update the `rev` field in `Cargo.toml` to a new commit and re-run the test suite with `--features rkyv` to confirm behavior.

Enabling rkyv and unchecked fast path
-------------------------------------

By default rkyv is optional behind the `rkyv` Cargo feature. Enable it with:

```sh
cargo test --features rkyv
```

For trusted inputs there's an additional (unsafe) unchecked zero-copy path exposed behind the `rkyv_unchecked` feature. This skips validation and performs an unchecked archived pointer conversion; only enable it when you fully trust the input bytes (for maximum speed). To run tests with the unchecked path enabled:

```sh
cargo test --features "rkyv rkyv_unchecked"
```

Benchmarking note
-----------------

The repository includes a small benchmark/test that compares the validated and unchecked rehydration paths. If you want to measure performance locally, run the bench (or the provided test) and compare timings. Be sure to pin dependencies and use the same CPU frequency/power settings for reproducible numbers.
