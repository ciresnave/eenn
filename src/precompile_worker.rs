use crate::{ExecutionContext, KernelCache};
use anyhow::Context;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::{Notify, Semaphore, mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::task::JoinSet;

/// A small cloneable cancellation token.
#[derive(Clone, Debug)]
pub struct CancellationToken {
    inner: Arc<CancellationInner>,
}

#[derive(Debug)]
struct CancellationInner {
    cancelled: AtomicBool,
    notify: Notify,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(CancellationInner {
                cancelled: AtomicBool::new(false),
                notify: Notify::new(),
            }),
        }
    }

    /// Trigger cancellation. Idempotent.
    pub fn cancel(&self) {
        if !self.inner.cancelled.swap(true, Ordering::SeqCst) {
            self.inner.notify.notify_waiters();
        }
    }

    /// Returns true if already cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }

    /// Async wait until cancelled. Returns immediately if already cancelled.
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        self.inner.notify.notified().await;
    }
}

#[derive(Debug)]
pub struct CompileTask {
    pub fingerprint: String,
    pub source: Vec<u8>,
    pub cancel: CancellationToken,
    /// Optional notify that will be signalled when the task actually starts compiling.
    pub started: Option<Arc<Notify>>,
}

impl Clone for CompileTask {
    fn clone(&self) -> Self {
        Self {
            fingerprint: self.fingerprint.clone(),
            source: self.source.clone(),
            cancel: self.cancel.clone(),
            started: self.started.clone(),
        }
    }
}

impl CompileTask {
    /// Convenience helper to create a task with a `Notify` that will be signalled
    /// when the worker actually starts compiling the task. Returns the task,
    /// the `Arc<Notify>` and a `CancellationToken` (also stored inside the task)
    /// so callers can cancel the task after it's started.
    pub fn with_started<Fp, S>(fingerprint: Fp, source: S) -> (Self, Arc<Notify>, CancellationToken)
    where
        Fp: Into<String>,
        S: Into<Vec<u8>>,
    {
        let notify = Arc::new(Notify::new());
        let token = CancellationToken::new();
        let task = CompileTask {
            fingerprint: fingerprint.into(),
            source: source.into(),
            cancel: token.clone(),
            started: Some(notify.clone()),
        };
        (task, notify, token)
    }
}

/// Handle for the running precompile service. Use `submit` to enqueue tasks and
/// `shutdown()` to gracefully stop the background worker and wait for in-flight compiles.
/// A function that performs compilation. It receives the CompileTask and returns a
/// Future resolving to the compiled artifact bytes or an error.
pub type CompileFn = dyn Fn(CompileTask) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<u8>>> + Send>>
    + Send
    + Sync;

pub struct CancelHandle {
    // registry keyed by id for scalable cancellation
    inner: Arc<AsyncMutex<std::collections::HashMap<usize, CancellationToken>>>,
}

impl CancelHandle {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AsyncMutex::new(std::collections::HashMap::new())),
        }
    }

    pub async fn register(&self, token: &CancellationToken) {
        let mut g = self.inner.lock().await;
        // use token address as a simple id; alternatively a UUID could be generated
        let id = Arc::as_ptr(&token.inner) as usize;
        g.insert(id, token.clone());
    }

    /// Cancel all registered tokens.
    pub async fn cancel_all(&self) {
        let g = self.inner.lock().await;
        for (_k, t) in g.iter() {
            t.cancel();
        }
    }
}

impl Clone for CancelHandle {
    fn clone(&self) -> Self {
        CancelHandle {
            inner: self.inner.clone(),
        }
    }
}

pub struct PrecompileService {
    tx: mpsc::Sender<(CompileTask, Option<oneshot::Sender<anyhow::Result<()>>>)>,
    worker_handle: JoinHandle<()>,
    /// Current approximate queue length (tasks enqueued but not yet taken by worker)
    pub queue_len: Arc<AtomicUsize>,
    /// Current number of in-flight compile tasks
    pub in_flight: Arc<AtomicUsize>,
    pub cancel_handle: CancelHandle,
}

impl PrecompileService {
    pub async fn submit(&self, task: CompileTask) -> anyhow::Result<()> {
        // increment queue length approximation, will be decremented by worker when taken
        self.queue_len.fetch_add(1, Ordering::SeqCst);
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send((task, Some(resp_tx)))
            .await
            .context("send task")?;
        // resp_rx carries the result of the compile; forward the error if any
        match resp_rx.await.context("await response")? {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Request graceful shutdown and wait for background worker to finish.
    pub async fn shutdown(self) {
        // Dropping self.tx will close the channel and cause the worker loop to exit
        drop(self.tx);
        // Await worker completion
        let _ = self.worker_handle.await;
    }

    /// Return a snapshot of current metrics.
    pub fn metrics_snapshot(&self) -> (usize, usize) {
        (
            self.queue_len.load(Ordering::SeqCst),
            self.in_flight.load(Ordering::SeqCst),
        )
    }
}

/// Spawn a background precompile worker.
///
/// - `cache` is cloned (must be Arc) so multiple workers can share it.
/// - `ctx` is ExecutionContext used for profiler events.
/// - `concurrency_limit` caps concurrent compile tasks.
pub fn spawn_precompile_worker(
    cache: Arc<tokio::sync::Mutex<KernelCache>>,
    ctx: Arc<ExecutionContext>,
    concurrency_limit: usize,
    compile_fn: Arc<CompileFn>,
) -> PrecompileService {
    let (tx, mut rx) =
        mpsc::channel::<(CompileTask, Option<oneshot::Sender<anyhow::Result<()>>>)>(64);
    let sem = Arc::new(Semaphore::new(concurrency_limit));
    let queue_len = Arc::new(AtomicUsize::new(0));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let cancel_handle = CancelHandle::new();

    // Worker main loop: owns a JoinSet for tracking per-compile tasks
    let qlen = queue_len.clone();
    let inflight = in_flight.clone();
    let ch = cancel_handle.clone();

    let worker_handle = tokio::spawn(async move {
        let mut joinset: JoinSet<()> = JoinSet::new();

        while let Some((task, resp)) = rx.recv().await {
            // taken from queue
            qlen.fetch_sub(1, Ordering::SeqCst);

            let permit = match sem.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => break, // semaphore closed
            };

            let cache = cache.clone();
            let ctx = ctx.clone();

            // register token with group cancel handle
            let _ = ch.register(&task.cancel).await;

            // increment in-flight
            inflight.fetch_add(1, Ordering::SeqCst);

            // spawn a task per compile (tracked in joinset)
            let cf = compile_fn.clone();
            let inflight_clone = inflight.clone();
            joinset.spawn(async move {
                if let Some(p) = &ctx.profiler {
                    p.record_event("compile_start", 1);
                }
                if let Some(n) = &task.started {
                    n.notify_waiters();
                }
                // Call provided compile function; race it against cancellation
                let compile_future = (cf)(task.clone());
                let compiled = tokio::select! {
                    biased;
                    _ = task.cancel.cancelled() => {
                        // cancelled
                        if let Some(p) = &ctx.profiler {
                            p.record_event("compile_cancelled", 1);
                        }
                        Err(anyhow::anyhow!("cancelled"))
                    }
                    r = compile_future => r,
                };

                // Write to cache if compiled
                let res = match compiled {
                    Ok(bytes) => {
                        let mut guard = cache.lock().await;
                        let write_res = guard.write_artifact(&task.fingerprint, &bytes);
                        if write_res.is_ok() {
                            if let Some(p) = &ctx.profiler {
                                p.record_event("compile_done", 1);
                            }
                        }
                        write_res
                    }
                    Err(e) => Err(e),
                };

                // respond if requested
                if let Some(r) = resp {
                    let _ = r.send(res.map_err(|e| e));
                }
                // decrement in-flight
                inflight_clone.fetch_sub(1, Ordering::SeqCst);
                drop(permit); // release semaphore permit
            });

            // Note: we don't aggressively poll joinset here; spawned tasks are still tracked
        }

        // Channel closed: wait for all spawned compile tasks to finish
        while let Some(_r) = joinset.join_next().await {}
    });

    PrecompileService {
        tx,
        worker_handle,
        queue_len,
        in_flight,
        cancel_handle,
    }
}
