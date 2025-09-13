#![cfg(feature = "tokio")]

use eenn::{
    CancellationToken, CompileTask, ExecutionContext, InMemoryProfiler, KernelCache,
    spawn_precompile_worker,
};
use std::sync::Arc;

#[tokio::test]
async fn started_notify_allows_deterministic_cancel() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let cache = KernelCache::new(dir.path()).expect("cache");
    let cache = Arc::new(tokio::sync::Mutex::new(cache));

    let profiler = Arc::new(InMemoryProfiler::new());
    let ctx = Arc::new(ExecutionContext::with_profiler(profiler.clone()));

    // compile function that waits a bit so we can cancel after start
    let compile_fn: std::sync::Arc<eenn::CompileFn> =
        std::sync::Arc::new(|task: eenn::CompileTask| {
            Box::pin(async move {
                // simulate work; cancellation should have been requested right after start
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
                let mut out = b"compiled:async:".to_vec();
                out.extend_from_slice(task.fingerprint.as_bytes());
                Ok(out)
            })
        });

    let service = spawn_precompile_worker(cache.clone(), ctx.clone(), 1, compile_fn);

    // create a task with a started notify and cancel token
    let (task, started, token) = CompileTask::with_started("deterministic", vec![]);

    // spawn a background task that runs the submit and sends the service back when done
    let (svc_tx, svc_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let res = service.submit(task).await;
        // send back the service along with the result so the test can continue and shutdown
        let _ = svc_tx.send((res, service));
    });

    // wait until worker signals it started
    started.notified().await;

    // now cancel the task
    token.cancel();

    // receive the result and the service back from the spawned task
    let (res, service) = svc_rx.await.expect("service spawn task dropped");
    let _ = res; // should be Err because compile was cancelled

    // ensure cache does not contain artifact
    let guard = cache.lock().await;
    assert!(guard.lookup("deterministic").is_none());

    // shutdown
    service.shutdown().await;
}
