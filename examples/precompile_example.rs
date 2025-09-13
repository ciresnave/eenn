use eenn::{
    CancellationToken, CompileTask, ExecutionContext, InMemoryProfiler, KernelCache,
    spawn_precompile_worker,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// Simple compile fn that just echoes the fingerprint after a short delay.
fn simple_compile_fn() -> Arc<eenn::CompileFn> {
    Arc::new(|task: CompileTask| {
        Box::pin(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            let mut out = b"example:compiled:".to_vec();
            out.extend_from_slice(task.fingerprint.as_bytes());
            Ok(out)
        }) as Pin<Box<dyn Future<Output = anyhow::Result<Vec<u8>>> + Send>>
    })
}

#[tokio::main]
async fn main() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let cache = KernelCache::new(dir.path()).expect("cache");
    let cache = Arc::new(tokio::sync::Mutex::new(cache));

    let profiler = Arc::new(InMemoryProfiler::new());
    let ctx = Arc::new(ExecutionContext::with_profiler(profiler.clone()));

    let compile_fn = simple_compile_fn();
    let service = spawn_precompile_worker(cache.clone(), ctx.clone(), 2, compile_fn);

    // submit a few tasks
    for i in 0..4usize {
        let token = CancellationToken::new();
        let task = CompileTask {
            fingerprint: format!("example-{}", i),
            source: vec![],
            cancel: token,
            started: None,
        };
        let _ = service.submit(task).await;
    }

    // gracefully shutdown
    service.shutdown().await;
    println!("example: done");
}
