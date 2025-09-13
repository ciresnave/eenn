#![cfg(feature = "tokio")]

use eenn::Profiler;
use eenn::{
    CancelHandle, CancellationToken, CompileFn, CompileTask, ExecutionContext, InMemoryProfiler,
    KernelCache, PrecompileService, spawn_precompile_worker,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// A compile fn that takes a bit longer to simulate load
fn heavy_compile_fn() -> Arc<CompileFn> {
    Arc::new(|task: CompileTask| {
        Box::pin(async move {
            // simulate heavier compile
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            let mut out = b"heavy:compiled:".to_vec();
            out.extend_from_slice(task.fingerprint.as_bytes());
            Ok(out)
        }) as Pin<Box<dyn Future<Output = anyhow::Result<Vec<u8>>> + Send>>
    })
}

#[tokio::test]
async fn shutdown_waits_for_inflight_under_load() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let cache = KernelCache::new(dir.path()).expect("cache");
    let cache = Arc::new(tokio::sync::Mutex::new(cache));

    let profiler = Arc::new(InMemoryProfiler::new());
    let ctx = Arc::new(ExecutionContext::with_profiler(profiler.clone()));

    let compile_fn = heavy_compile_fn();
    let service = spawn_precompile_worker(cache.clone(), ctx.clone(), 4, compile_fn);

    // submit many tasks
    for i in 0..16usize {
        let token = CancellationToken::new();
        let task = CompileTask {
            fingerprint: format!("heavy-{}", i),
            source: vec![],
            cancel: token,
            started: None,
        };
        // don't await
        let _ = service.submit(task).await;
    }

    // shutdown should wait until in-flight tasks finish
    service.shutdown().await;

    // All tasks either compiled or were cancelled; ensure no panic and some work done
    let snap = profiler.snapshot();
    assert!(snap.events.get("compile_done").copied().unwrap_or(0) >= 1);
}
