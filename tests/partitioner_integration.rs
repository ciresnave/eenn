use std::sync::Arc;
use std::time::Instant;

use eenn::{
    Calibration, ExecutionContext, GreedyPartitioner, InMemoryProfiler, KernelCache, Partitioner,
    Profiler,
};

// Test-only Planner that uses GreedyPartitioner to create ExecutionPlans and
// a stubbed precompile worker that writes quick artifacts into the KernelCache.
#[test]
fn planner_and_precompile_worker_integration() {
    // Create a set of mock ops (reuse types from the library tests via local definitions)

    // Create a kernel cache in tempdir and persist calibration into the cache root
    let dir = tempfile::tempdir().expect("tmpdir");
    let calib = Calibration {
        bandwidth_bytes_per_sec: 500.0 * 1024.0 * 1024.0,
    }; // 500 MB/s
    calib
        .persist_to_cache_root(dir.path())
        .expect("persist calibration");

    // Create cache and then load calibration from the cache root
    let mut cache = KernelCache::new(dir.path()).expect("cache create");
    let loaded = Calibration::load_from_cache_root(dir.path())
        .expect("load calibration")
        .expect("calib present");
    let bytes_per_ms = loaded.bytes_per_ms();
    let lookahead_values = [0usize, 1usize, 2usize, 4usize];
    for &lookahead in &lookahead_values {
        let partitioner = GreedyPartitioner {
            lookahead,
            bytes_per_ms,
            max_ops_in_fusion: 128,
        };
        struct OpMock(&'static str, bool, f64, usize);
        impl eenn::partitioner::Op for OpMock {
            fn name(&self) -> &str {
                self.0
            }
            fn gpu_capable(&self) -> bool {
                self.1
            }
            fn estimate(&self) -> (f64, usize) {
                (self.2, self.3)
            }
        }

        let ops: Vec<Arc<dyn eenn::partitioner::Op>> = vec![
            Arc::new(OpMock("a", true, 2.0, 0)),
            Arc::new(OpMock("b", true, 2.0, 0)),
            Arc::new(OpMock("c", true, 0.5, 1000)),
            Arc::new(OpMock("d", false, 0.0, 0)),
        ];

        // Partition
        let segments = partitioner.partition(&ops);
        assert!(
            !segments.is_empty(),
            "Expected at least one GPU segment for lookahead={}",
            lookahead
        );

        // Set up profiler + execution context
        let profiler = std::sync::Arc::new(InMemoryProfiler::new());
        let ctx = ExecutionContext::with_profiler(profiler.clone());

        // In-test simple cache stats
        let mut misses = 0usize;

        // For each segment, compute a fake fingerprint and check cache; if missing, simulate compile and write artifact
        for seg in &segments {
            let fp = format!("seg-{}-{}", seg.start, seg.end);
            if cache.lookup(&fp).is_none() {
                misses += 1;
                if let Some(p) = &ctx.profiler {
                    p.record_event("cache_miss", 1);
                }
                // simulate compile time (quick tier)
                std::thread::sleep(std::time::Duration::from_millis(1));
                let artifact = format!("compiled:generic:{}", fp).into_bytes();
                cache.write_artifact(&fp, &artifact).expect("write");
            } else if let Some(p) = &ctx.profiler {
                p.record_event("cache_hit", 1);
            }
            let bytes = cache.read_artifact(&fp).expect("read");
            assert!(bytes.starts_with(b"compiled:"));

            // Simulate metrics and hotness scoring
            #[derive(Clone)]
            struct ArtifactMetrics {
                use_count: usize,
                last_used_at: std::time::Instant,
                avg_runtime_ms: f64,
                compile_time_ms: f64,
            }

            // fake metrics
            let mut metrics = ArtifactMetrics {
                use_count: 10,
                last_used_at: Instant::now(),
                avg_runtime_ms: 2.0,
                compile_time_ms: 5.0,
            };

            fn compute_hotness(metrics: &ArtifactMetrics, now: Instant) -> f64 {
                let age_s = (now.duration_since(metrics.last_used_at).as_secs_f64()).max(1.0);
                let time_decay =
                    (-0.1 * (now.duration_since(metrics.last_used_at).as_secs_f64())).exp();
                let usage_velocity = metrics.use_count as f64 / age_s;
                let avg_saved_ms = (metrics.avg_runtime_ms - 0.5).max(0.0); // assume jit overhead 0.5ms
                let compile_roi = if metrics.compile_time_ms > 0.0 {
                    (metrics.use_count as f64 * avg_saved_ms) / metrics.compile_time_ms
                } else {
                    0.0
                };
                usage_velocity * compile_roi * time_decay
            }

            // Evaluate hotness and optionally promote to specialized
            let now = Instant::now();
            let hotness = compute_hotness(&metrics, now);
            // promotion threshold is arbitrary for the test
            if hotness >= 0.1 {
                // simulate heavy specialized compile and atomic replacement
                if let Some(p) = &ctx.profiler {
                    p.record_event("promotion_start", 1);
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
                let specialized = format!("compiled:specialized:{}", fp).into_bytes();
                cache
                    .write_artifact(&fp, &specialized)
                    .expect("write specialized");
                if let Some(p) = &ctx.profiler {
                    p.record_event("promotion_done", 1);
                }
            } else if let Some(p) = &ctx.profiler {
                p.record_event("promotion_skipped", 1);
            }

            // Simulate demotion if artifact cools off: reduce use_count and recompute
            metrics.use_count = 0; // cooled off
            let hotness2 = compute_hotness(&metrics, Instant::now());
            if hotness2 < 0.01 {
                // demote: rewrite generic artifact
                let generic = format!("compiled:generic:{}", fp).into_bytes();
                cache.write_artifact(&fp, &generic).expect("write demoted");
                if let Some(p) = &ctx.profiler {
                    p.record_event("demoted", 1);
                }
            }
        }

        // Validate profiler recorded events for this iteration
        let snap = profiler.snapshot();
        assert_eq!(
            snap.events.get("cache_miss").copied().unwrap_or(0) as usize,
            misses
        );
        // Either we started promotions for some segments or we explicitly skipped them
        let promotion_start = snap.events.get("promotion_start").copied().unwrap_or(0);
        let promotion_skipped = snap.events.get("promotion_skipped").copied().unwrap_or(0);
        assert!(
            promotion_start > 0 || promotion_skipped > 0,
            "expected promotion_start or promotion_skipped to be recorded"
        );
        // Ensure demotion path exercised at least once (we force cooled-off metrics in test)
        let demoted = snap.events.get("demoted").copied().unwrap_or(0);
        assert!(demoted > 0, "expected at least one demotion event");
    }
}
