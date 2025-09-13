#![cfg(feature = "tokio")]

use eenn::{
    CancellationToken, CompileTask, ExecutionContext, InMemoryProfiler, KernelCache, Profiler,
    spawn_precompile_worker,
};
use std::sync::Arc;

#[tokio::test]
async fn precompile_worker_throttle_and_cancel() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let cache = KernelCache::new(dir.path()).expect("cache");
    let cache = Arc::new(tokio::sync::Mutex::new(cache));

    let profiler = Arc::new(InMemoryProfiler::new());
    let ctx = Arc::new(ExecutionContext::with_profiler(profiler.clone()));

    // spawn with concurrency 2; provide a trivial compile function
    let compile_fn: std::sync::Arc<eenn::CompileFn> =
        std::sync::Arc::new(|task: eenn::CompileTask| {
            Box::pin(async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                let mut out = b"compiled:async:".to_vec();
                out.extend_from_slice(task.fingerprint.as_bytes());
                Ok(out)
            })
        });

    let service = spawn_precompile_worker(cache.clone(), ctx.clone(), 2, compile_fn);

    // Submit 4 tasks, cancel one
    let mut cancels: Vec<Option<CancellationToken>> = vec![];
    for i in 0..4usize {
        let fp = format!("task-{}", i);
        let token = CancellationToken::new();
        cancels.push(Some(token.clone()));
        let task = CompileTask {
            fingerprint: fp.clone(),
            source: vec![],
            cancel: token,
            started: None,
        };
        // submit and don't await response to exercise internal queueing
        let _ = service.submit(task).await;

        // If this is the task we intend to cancel, cancel immediately to increase chance it aborts
        if i == 2 {
            if let Some(t) = cancels.last_mut().and_then(|o| o.take()) {
                t.cancel();
            }
        }
    }

    // give time for workers to run
    // give time for workers to run (must be larger than the simulated compile time)
    tokio::time::sleep(tokio::time::Duration::from_millis(400)).await;

    // Now assert cache contains compiled artifacts for tasks. Cancellation timing is racy
    let guard = cache.lock().await;
    let mut compiled_count = 0usize;
    for i in 0..4usize {
        let fp = format!("task-{}", i);
        if guard.lookup(&fp).is_some() {
            compiled_count += 1;
        }
    }

    // We expect at least 3 successful compilations (one cancellation at most)
    assert!(
        compiled_count >= 3,
        "expected at least 3 compiled artifacts, got {}",
        compiled_count
    );

    // Check profiler events: either a cancellation was recorded, or all 4 compiled (race)
    let snap = profiler.snapshot();
    let cancelled = snap.events.get("compile_cancelled").copied().unwrap_or(0);
    if compiled_count < 4 {
        // if one is missing, ensure cancellation was recorded
        assert!(
            cancelled >= 1,
            "expected cancellation event when fewer than 4 compiled"
        );
    }

    // gracefully shutdown the service
    service.shutdown().await;
}
