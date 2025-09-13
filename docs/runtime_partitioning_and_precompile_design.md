# Runtime Partitioning, Fusion, and Adaptive Precompilation Design

This document describes a practical, hybrid JIT/AOT design for runtime partitioning, operator fusion, caching, and background precompilation for the EENN project. It covers motivations, architecture, fingerprinting, caches, triggers, background compilation workflows, eviction strategies, and an implementation roadmap.

Goals

- Allow pipelines (stacks of ops) to run on CPU or GPU without rewriting operator definitions.
- Fuse ops into microkernels to reduce kernel launch and memory traffic overheads.
- Support runtime compilation (JIT fusion) for flexibility and AOT precompilation for deterministic performance and devices without runtime compilation.
- Maintain a multi-level cache (in-memory, on-disk, remote) and a background precompile worker that adaptively precompiles hot artifacts.

---

## 1. Overview and Key Concepts

Op (logical operator): single-source algorithmic specification. Every op provides:

- a CPU implementation (native Rust/ndarray SIMD-friendly)
- metadata and a GPU codegen hook (WGSL snippet or IR generator)
- device support flags and cost hints

Partitioner: at runtime, partitions a pipeline (linear or DAG) into segments that should run on CPU vs GPU. For GPU-capable segments, it attempts to fuse them into a single microkernel.

Fusion: composing operator-level snippets into a single kernel body (WGSL or IR) so a single dispatch performs the computation for many ops.

Linking: composing already-compiled microkernels with light glue — cheaper than fusion but may leave intermediates in device memory.

Kernel cache: persistent storage for fused kernels and/or linked microkernels. Keys are canonical fingerprints derived from op sequences, shapes, params, target features, and codegen versions.

Background precompile worker: monitors cache usage, detects hot artifacts, and compiles AOT/specialized binaries in the background (progressively).

---

## 2. Partitioner Strategy (Hybrid JIT + Optional AOT)

- Runtime partitioner is the default. It adapts to input sizes and device capabilities.
- For each pipeline invocation:

  1. Query op metadata and input shapes.
  2. Walk the pipeline greedily (initial approach): accumulate fusable ops that are GPU-capable and whose estimated GPU benefit (compute - transfer) is positive.
  3. For each candidate GPU segment, compute a fingerprint and check kernel cache. If cached, use cached kernel; otherwise, compose WGSL/IR and compile (JIT) or schedule background compilation.

- For stable models/pipelines, use a separate AOT tool that runs the same partitioner offline and precompiles artifacts into the same cache format.

Pluggable partitioner

Start with a simple greedy partitioner, but explicitly design a pluggable `Partitioner` interface so we can swap in more sophisticated strategies later (lookahead, dynamic programming, reinforcement-learned policies).

Example trait (sketch):

```rust
pub trait Partitioner: Send + Sync {
    fn partition(&self, ops: &[Box<dyn Op>], ctx: &PartitionCtx) -> Vec<Segment>;
}

pub struct GreedyPartitioner { pub lookahead: usize }
```

Notes:

- `lookahead = 0` => pure greedy.
- Small `lookahead` windows allow evaluating a few partitioning choices without the cost of full DP.

DP fallback and tuning

- For small or medium-sized subgraphs (for example, N <= 64 ops or when estimated cost variance is high), the planner may run a windowed dynamic-programming (DP) partitioner to find a globally cheaper partition for that window. This gives a compromise between pure greedy speed and global optimality.
- DP should be restricted by node-count and wall-time budget (e.g., max 10ms) to avoid pathological compile-time blowup.

DP sketch (windowed):

```rust
// Run DP on a sliding window of ops of size W. For each window, compute optimal partitioning
// using cost(i,j) estimates; then accept local improvements subject to a global budget.
fn windowed_dp_partition(ops: &[Op], window: usize, budget_ms: u64) -> Vec<Segment> { /* ... */ }
```

Tuning `lookahead`

- Default: `lookahead = 2` (small, low overhead). Allow runtime config `PARTITION_LOOKAHEAD` to tune.
- Heuristic: increase lookahead when many ops are GPU-capable and data-transfer-to-compute ratio low.

---

## 3. Memory & Intermediate Lifetimes

Managing intermediates is critical on memory-constrained devices. The partitioner and planner must track tensor lifetimes and use a memory pool to reuse device buffers.

Execution planner data:

```rust
pub struct TensorLifetime { pub tensor_id: usize, pub first_use: usize, pub last_use: usize }
pub enum MemoryPoolStrategy { BinnedAllocator, Buddy, Arena }

pub struct ExecutionPlan {
    pub segments: Vec<Segment>,
    pub intermediate_lifetimes: Vec<TensorLifetime>,
    pub memory_pool: MemoryPoolStrategy,
}
```

Allocator notes:

- Use a lightweight bucketed/binned allocator for common tensor sizes.
- Optionally allow spilling to host memory with an IO budget for memory-constrained GPUs.

Allocator defaults and defragmentation

- Default allocator: `BinnedAllocator` (balance of simplicity and low fragmentation). Make allocator pluggable via runtime config.
- For long-running sessions, include an optional defragmentation pass that attempts to compact free ranges and coalesce small free blocks. Defragmentation must be safe (no live pointers) and run during low-load windows.

Example allocator usage (sketch)

```rust
// Simplified binned allocator interface sketch used by the planner to request device buffers
pub struct BinnedAllocator { /* ... */ }

impl BinnedAllocator {
  pub fn alloc(&mut self, size: usize) -> DeviceBufferRef { /* allocate or reuse */ unimplemented!() }
  pub fn free(&mut self, buf: DeviceBufferRef) { /* return to pool */ }
}

// Planner example: reserve tensors for intermediate lifetimes
let mut allocator = BinnedAllocator::new();
for lifetime in &plan.intermediate_lifetimes {
  let bytes = estimate_bytes_for_tensor(&lifetime);
  let buf = allocator.alloc(bytes);
  assign_buffer_to_tensor(lifetime.tensor_id, buf);
}
```

---

## 4. Fingerprinting, Versioning, and Canonicalization

Purpose

- Produce deterministic keys for cached artifacts so identical fused subgraphs map to the same compiled artifact.

Fingerprint components (extensible):

- Ordered op sequence: for each op include name, op-impl-version, static params
- Shapes (or shape class) and dtypes
- Device feature set (sorted list)
- Codegen/composer version
- Optional toolchain/compiler version

Fingerprint format and versioning

- Embed a `version` field in the fingerprint manifest to allow safe evolution of the fingerprinting rules.
- Store the canonical JSON manifest alongside the artifact for human inspection and debugging.

Example fingerprint manifest (JSON sketch):

```json
{
  "version": 1,
  "ops": [ { "name": "relu", "impl": "v1" }, { "name": "scale", "impl": "v2", "params": {"w":0.5} } ],
  "shapes": { "in0": [1,64], "out0": [1,64] },
  "dtype": "f32",
  "device_features": ["float32", "workgroup_size_256"],
  "composer_version": "wgsl-composer-0.1",
  "toolchain": "rustc-1.70"
}
```

Extensible fingerprint components (Rust sketch):

```rust
pub trait FingerprintComponent { fn contribute_to_hasher(&self, hasher: &mut Sha256); }

pub struct Fingerprint { pub version: u32, pub hash_hex: String, pub manifest: serde_json::Value }
```

Canonicalization & Signing Verification (guidance)

- Canonicalization rules:
  - Always sort object keys lexicographically when computing the canonical JSON.
  - Normalize floating-point numbers used in manifest fields to a fixed textual representation (for example, IEEE hex or a fixed decimal precision) to avoid accidental mismatches from formatting differences.
  - For optional fields, prefer explicit null values instead of omitting them; this keeps manifests structurally stable.
  - For arrays (like device_features) sort entries or document an explicit canonical order.

- Tests to add:
  - Unit tests that serialize the same manifest with shuffled keys and differing float formatting and assert the canonical JSON and fingerprint are identical.
  - Fuzz tests that generate manifest variants to ensure canonicalization is stable across inputs.

- Remote cache signing & verification:
  - Recommended signature scheme: ed25519.
  - Manifest fields: include `signed_by` (signer id) and `signature` (base64-ed25519(manifest_bytes)).
  - Verification flow: when loading a remote artifact, verify signature against a local trust store of public keys before accepting binaries or placing them in the cache.
  - Trust store and rotation: maintain a signed rotation manifest locally; reject signatures by unknown or revoked keys.

---

## 5. Kernel Cache Design

Storage layers

- In-memory LRU cache for the hot set (fast lookups).
- On-disk persistent cache in `cache_root/artifacts/` with per-artifact metadata files.
- Optional remote/registry for distribution.

What to store

- Source (WGSL/IR), compiled binary(s) per-device (if available), manifest (metadata), metrics (use_count, last_used_at, avg_runtime_ms, compile_time_ms), artifact kind (generic vs specialized).

Atomic writes and manifest index

- Write compiled artifacts to a temp file and atomically rename into place.
- Maintain a top-level index mapping fingerprints to metadata to accelerate listing and eviction. Index updates must be atomic.

Specialization tag

- `specialization.kind = "generic" | "static"`
- For `static` artifacts include the exact `shape` field; for `generic` artifacts omit shape or include a shape class.

---

Eviction policies and tuning

- Default eviction: hybrid LRU/LFU. Keep an in-memory LRU for fast lookups and an LFU counter stored in the manifest metadata to protect frequently-used artifacts from short-lived bursts.
- Cost-aware eviction: compute an eviction score per artifact that factors size, last_used_at, use_count, compile_time_ms, and a device-locality multiplier. Evict the artifact with the lowest score.
- Tunables: expose knobs `cache.eviction_policy` (lru | lfu | cost_aware), `cache.max_bytes`, and `cache.margin_bytes` to control behavior. For remote caches, respect a readonly flag and only fetch artifacts rather than evicting local copies.

Example eviction score sketch:

```rust
fn eviction_score(meta: &ArtifactMeta, now: Instant) -> f64 {
  let age_s = (now - meta.last_used_at).as_secs_f64().max(1.0);
  let freq = meta.use_count as f64 / age_s; // recent frequency
  let size_penalty = meta.size_bytes as f64 / 1e6; // MB
  let compile_cost = meta.compile_time_ms.max(1.0);
  (freq * 100.0) / (size_penalty * compile_cost)
}
```

## 6. Hotness Scoring, Trigger Policy, and Temporal Decay

Metrics to collect

- first_seen_at, last_used_at
- use_count, cumulative_runtime_ms, avg_runtime_ms
- jit_compile_time_ms, precompile_state
- artifact_size_bytes, residency_time

Example time-decayed hotness score (sketch):

```rust
fn compute_hotness(metrics: &ArtifactMetrics, now: Instant) -> f64 {
    let age_s = (now - metrics.first_seen_at).as_secs_f64().max(1.0);
    let time_decay = (-0.1 * (now - metrics.last_used_at).as_secs_f64()).exp();
    let usage_velocity = metrics.use_count as f64 / age_s;
    let avg_saved_ms = (metrics.avg_runtime_ms - metrics.jit_compile_overhead_ms).max(0.0);
    let compile_roi = if metrics.compile_time_ms > 0.0 {
        (metrics.use_count as f64 * avg_saved_ms) / metrics.compile_time_ms
    } else { 0.0 };
    usage_velocity * compile_roi * time_decay
}
```

Trigger policy notes

- Precompile when score >= threshold and `precompile_state == none` and resources permit.
- Use a hysteresis window to avoid flip-flopping (promotion/demotion thresholds).

---

## 7. Background Precompile Worker and Cancellation

Goals

- Compile hot kernels progressively (quick then tuned) while respecting CPU/GPU usage budgets.
- Abort or demote jobs if artifact cools off or resources are needed by foreground tasks.

Worker behavior

- Use a priority queue ordered by hotness score.
- Support cancellation tokens (e.g., `tokio_util::sync::CancellationToken`) so a running compile can be aborted cleanly.
- Write quick artifacts first; then optionally run a heavy specialized compile and atomically replace the artifact on success.

Worker throttling policy

- Periodically sample system load (CPU utilization, GPU queue depth, free GPU memory). If CPU > 80% or GPU free memory < 10% or GPU queue length above threshold, throttle worker concurrency or pause until conditions improve.
- Expose runtime knobs to limit worker resource use (max_threads, max_gpu_memory_bytes) and integrate with job priority: foreground tasks always have higher priority.

Chaos & robustness testing

- Add chaos tests that simulate:
  - missing or corrupted manifest files (ensure validation and graceful fallback to JIT)
  - partially written artifacts during crash (atomic rename semantics should prevent exposing partial artifacts)
  - background worker crashes and restart behavior
  - signature verification failures for remote artifacts (must reject and fall back)

Cancellation / demotion rules

- If artifact score drops below demote threshold while compiling, set `precompile_state = demoted` and cancel further work.
- If the compile fails, record failure and back off (exponential backoff) before re-trying.

---

## 8. Error Handling, Fallbacks, and Execution Strategy

Define explicit execution strategies and fallbacks so the runtime can recover from compile/execution failures.

```rust
pub enum ExecutionStrategy {
    PreferGPU { cpu_fallback: bool },
    PreferCPU { gpu_fallback: bool },
    Adaptive { failure_threshold: u32 },
}
```

Failure modes to handle

- GPU compilation failure: fall back to CPU execution (if available) and mark the artifact with `last_compile_failed`.
- Device OOM at runtime: retry with smaller batch or fall back to CPU; optionally spill intermediates to host.
- Background worker crash: supervisor restarts worker and increments failure counters in metadata.

---

## 9. Fusion vs Linking and Memory Tradeoffs

- Fusion (inline ops) is preferred for short elementwise chains to reduce memory traffic.
- Linking is cheaper if reuse of compiled microkernels is frequent or fusion cost is high.
- The planner must weigh memory pressure from intermediates against kernel launch overhead.

Fusion vs Linking heuristic

- Quantitative threshold: prefer fusion when estimated_global_memory_saved_bytes / (compile_time_ms + epsilon) >= FUSE_FACTOR, where FUSE_FACTOR is a tunable constant (e.g., 1000 bytes per ms). This compares memory-bandwidth savings to compile cost.
- Start with FUSE_FACTOR = 1000 and tune with benchmarks. The planner must also cap fusion size (max_ops_in_fusion, default 128 ops or 10k lines WGSL) to avoid runaway compile times.

---

## 10. Cost Model & Calibration

Cost estimate shape

```rust
pub struct CostEstimate {
    pub compute_secs: f64,
    pub memory_bytes: usize,
    pub energy_score: Option<f32>,
    pub cache_locality: f32,
}
```

Calibrate device constants with a microbenchmark on first-run (bandwidth, per-workgroup latency, flops). Store calibration in the cache root so results persist.

Calibration & feedback loop

- Run a short microbenchmark on first-run and periodically (or on demand) to estimate: memory bandwidth (host<->device and device-local), per-kernel launch latency, and device FLOPS.
- Persist calibration in the cache root along with a calibration version; invalidate cached artifacts that depend on incompatible calibration versions.
- Add a feedback loop: record actual runtime vs estimated runtime for segments and adapt cost-model parameters over time (e.g., via simple exponential moving average updates).

## 11. Testing & Validation Strategy

- Unit tests for fingerprint canonicalization and manifest parsing (already implemented for canonical JSON).
- Property-based tests for the partitioner: generate random operator sequences, shapes, and parameters then assert CPU sequential execution equals partitioned+fused execution (within numeric tolerance).

Example test plan

- `proptest` generates small DAGs and elementwise ops; compare outputs and check memory lifetimes.
- Stress tests for memory pooling and spill-to-host behavior under limited device memory.
- Integration test that simulates repeated runs to validate precompile promotion and that precompiled artifacts reduce execution latency.

---

## 12. Monitoring, Profiling & Observability

- Expose: cache hit/miss, precompile queue size, precompile job stats, compile_time histograms, cache disk usage.
- Add profiler hooks in `ExecutionContext` and record per-dispatch runtime, compile times, and memory peaks.

Profiler sketch

```rust
pub trait Profiler: Send + Sync { fn record_event(&self, name: &str, micros: u64); fn snapshot(&self) -> ProfilerSnapshot; }

pub struct ExecutionContext { pub profiler: Option<Arc<dyn Profiler>>, pub debug_mode: bool, pub artifact_provenance: HashMap<String, ArtifactSource>, }
```

Artifact provenance

- Record whether an artifact was generated by quick-JIT, tuned-AOT, or pulled from remote registry. This helps debugging and tracing performance regressions.

---

## 13. Device Topology Awareness

- Consider multi-GPU / NUMA / unified-memory systems. Device selection should account for interconnect bandwidth and device affinity.
- Planner should expose `DeviceInfo` (memory_capacity, bandwidth, compute, peer_access) used by the cost model.

Topology-aware scheduling

- Query device topology at startup (using platform APIs) and expose a `DeviceGraph` representing peer links and bandwidth.
- Prefer devices with direct peer links (e.g., NVLink) for large pipelines; prefer local-device placement to minimize host-device transfers.

Device topology implementation notes

- Use platform APIs (Vulkan/DirectX/OpenCL/Metal) or `wgpu` adapter info to query device properties such as memory size, unified-memory support, and peer-access capability. Cache these details in `DeviceInfo` at startup.
- Build a `DeviceGraph` where edges are annotated with estimated peer bandwidth and latency. Prefer scheduling large, transfer-heavy segments on device-pairs with high peer bandwidth.
- For multi-process or multi-node setups, support a pluggable discovery backend that can return topology info from the system (e.g., `nvidia-smi topo --matrix` on NVIDIA systems) or from a cluster manager.

Example `DeviceInfo` sketch:

```rust
pub struct DeviceInfo {
  pub id: String,
  pub memory_bytes: usize,
  pub unified_memory: bool,
  pub peer_links: Vec<(String /* peer id */, f64 /* GB/s */)>,
}
```

Binary manifest optimization

- For internal on-disk cache operations, optionally store manifests in a compact binary format (MessagePack or Cap'n Proto) to speed lookup and reduce parse overhead; still store canonical JSON alongside for debugging and signing.

---

## 14. Shape Specialization Strategy

- Support two artifact kinds:

  - `generic`: compiled code that accepts variable shapes (uses dynamic indexing / loops).
  - `static/specialized`: compiled for exact shapes; gives best performance for hot, stable pipelines.

- The background worker may produce a `static` artifact after observing sufficient stability and score.

---

## 15. Fingerprint / Cache Manifest Example and Fields

Fields to include for each artifact's manifest:

- `fingerprint_version`
- `ops` (sequence + impl versions)
- `shapes` or `shape_class`
- `dtype`
- `device_features`
- `composer_version`
- `artifact_kind` (`generic` | `static`)
- `metrics` (use_count, last_used_at, avg_runtime_ms)

Store both the canonical manifest and a human-friendly JSON index for debugging.

---

## 16. Implementation Roadmap (detailed)

1. Update design doc (this change) and finalize manifest schema.
2. Implement fingerprint generator + canonicalizer (done).
3. Implement kernel cache API and on-disk manifest layout (in-memory index + atomic writes).
4. Add lightweight metrics/profiler hooks and persistent calibration store.
5. Implement pluggable `Partitioner` API and default `GreedyPartitioner` (with optional lookahead).
6. Implement simple WGSL composer for elementwise fusion and a minimal JIT integration.
7. Integrate staging pool/executor and memory pooling / lifetime tracking.
8. Implement background precompile worker with cancellation and demotion logic.
9. Add AOT CLI to precompile stable pipelines and populate cache.
10. Add comprehensive tests: proptest for partitioner, integration tests for precompile triggers and cache behavior.

---

## Reviewer feedback & planned actions

Summary of reviewer response

- Strengths: hybrid JIT/AOT approach, modular pluggable components, careful fingerprinting/canonicalization design, explicit lifecycle and memory management, and a sound calibration/feedback plan.
- Risks called out: system complexity (many interacting subsystems), partitioner latency for large graphs, cache thrashing or disk-I/O pressure, background-worker resource contention, and potential cache explosion from shape specialization.

Practical next steps (prioritized)

1. Manifest & canonicalization validation (this sprint)

- Add unit tests that validate `docs/sample_signed_manifest.json` against `docs/manifest_schema.json` and assert canonicalization properties.
- Implement a small manifest validator test (see `tests/manifest_validator.rs`).

2. Signature verification helper (low-risk, high-value)

- Add a minimal ed25519 verification helper and a sketch of a local trust-store interface. This will be non-invasive and test-only initially.

3. Integration & observability (medium-term)

- Add end-to-end integration tests that exercise partitioner ⇄ cache ⇄ precompile worker interactions.
- Emit detailed metrics and logs for partitioning decisions, cache hits/misses, and worker activity to aid debugging.

4. Benchmarking the partitioner and cache (concurrent)

- Create microbenchmarks and synthetic workloads to profile partitioner latency and cache I/O under stress.

5. Prototype cache and greedy partitioner (next milestone)

- Implement a minimal on-disk kernel cache API (atomic writes + index) and a GreedyPartitioner with lookahead tuning.

Notes

- These steps follow the reviewer's recommendation to prototype core components (fingerprinting, cache, partitioner) and benchmark early. They prioritize low-risk validations and observability before adding heavy-weight DP or reinforcement approaches.
- I'll proceed with the manifest validator test now and run the test suite. After that completes, I can add the signature helper sketch unless you'd prefer a different priority.

If you'd like I can now:

- run a Markdown style/lint pass and fix spacing issues, or
- produce a compact manifest JSON Schema and a small example cache layout (file names, index format), or
- implement the on-disk kernel cache API next (requires code changes).

Which of these should I do next? (No runtime code changes unless you ask.)
