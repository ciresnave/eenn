use crate::tensor::{DType, DeviceBuffer};
use std::borrow::Cow;
use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
// wgpu util extension trait for create_buffer_init / make_spirv
use anyhow::Result;
use ndarray::ArrayD;
use once_cell::sync::OnceCell;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;
use wide::f32x4;

// Small SIMD helper: fill buf[i] = i as f32 * scale
fn fill_arange_mul(buf: &mut [f32], scale: f32) {
    let n = buf.len();
    let mut i = 0usize;
    let mut base = 0f32;
    // process in chunks of 4
    while i + 4 <= n {
        let idxs = f32x4::new([base, base + 1.0, base + 2.0, base + 3.0]);
        let mul = f32x4::splat(scale) * idxs;
        let arr = mul.to_array();
        buf[i] = arr[0];
        buf[i + 1] = arr[1];
        buf[i + 2] = arr[2];
        buf[i + 3] = arr[3];
        i += 4;
        base += 4.0;
    }
    while i < n {
        buf[i] = base * scale;
        i += 1;
        base += 1.0;
    }
}

// System analysis integration: probe system capabilities once in the
// background and cache an integer score that can be consulted by
// adapter selection heuristics. This is intentionally lightweight and
// non-blocking for initialization paths.
static SYS_ANALYSIS_SCORE: AtomicI32 = AtomicI32::new(0);
#[allow(dead_code)]
static SYS_ANALYSIS_STARTED: OnceCell<()> = OnceCell::new();

#[allow(dead_code)]
fn start_system_analysis_probe_once() {
    // Ensure we only spawn the background probe once.
    if SYS_ANALYSIS_STARTED.set(()).is_ok() {
        std::thread::spawn(|| {
            // Try to create a SystemAnalyzer and run its async analysis
            // on a background thread. We avoid calling internal `probe`
            // modules and instead use the public analyzer API if
            // available. Keep this robust to differing crate versions by
            // early-returning on any error.
            // Construct analyzer and run the async analysis synchronously
            // on this background thread. If the analyzer or analysis fails
            // we simply skip updating the cached score.
            let mut analyzer = system_analysis::SystemAnalyzer::new();
            if let Ok(profile) = pollster::block_on(async { analyzer.analyze_system().await }) {
                // Use the crate-provided aggregate GPU score when available.
                let g = profile.gpu_score() as i32;
                SYS_ANALYSIS_SCORE.store(g.min(i32::MAX as i32), Ordering::SeqCst);
            }
        });
    }
}

// system_analysis integration is intentionally omitted here to avoid
// introducing an optional dependency into this module; enabling the
// feature in the workspace can add integration higher-level if desired.

// Per-bucket cap for staging pool to avoid unbounded growth
const MAX_PER_BUCKET: usize = 4;

// Shorthand type for the staging pool shared across devices/guards
type StagingPool = Arc<Mutex<BTreeMap<NonZeroU64, VecDeque<StagingEntry>>>>;

/// RAII guard that owns a staging entry and returns it to the pool on Drop.
pub struct StagingGuard {
    pub key: NonZeroU64,
    pub entry: Option<StagingEntry>,
    pub pool: StagingPool,
    pub created_mapped: bool,
}

impl StagingGuard {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.entry.as_ref().expect("entry present").buffer
    }

    pub fn created_mapped(&self) -> bool {
        self.created_mapped
    }
}
/// The mapped view's lifetime is tied to this guard. On Drop the buffer
/// is unmapped and the staging entry is returned to the pool.
pub struct StagingMappedWriteGuard {
    guard: Option<StagingGuard>,
    // Owned staging bytes the caller may mutate. We avoid holding live
    // BufferViewMut objects across the guard lifetime to prevent
    // unmap/drop ordering issues with backend validation. The collected
    // bytes are written back into the staging buffer on Drop or when
    // take_guard() is called using the device queue.
    mapped: Option<Vec<u8>>,
    // Arc to the owning GpuDevice so Drop can access the queue to write
    // back mapped bytes. We use DeviceManager::global() to obtain an Arc
    // to the device when constructing the guard.
    device: Option<Arc<GpuDevice>>,
}

impl StagingMappedWriteGuard {
    /// Access the mapped range as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.mapped.as_deref_mut().expect("mapped range present")
    }

    /// Take ownership of the inner StagingGuard, preventing it from
    /// returning the entry to the pool on Drop. Caller receives the guard
    /// and must handle returning the entry.
    pub fn take_guard(mut self) -> StagingGuard {
        // Finalize any pending mapped bytes by writing them into the
        // staging buffer via the device queue, then return the underlying
        // StagingGuard so the caller can perform copies/submission.
        if let Some(data) = self.mapped.take() {
            if let Some(g) = self.guard.as_ref() {
                if let Some(dev) = self.device.as_ref() {
                    dev.queue.write_buffer(g.buffer(), 0, &data);
                    // Ensure the write has been processed before returning the
                    // guard; subsequent GPU copies may read from the staging
                    // buffer and would observe stale contents otherwise.
                    dev.device.poll(wgpu::Maintain::Wait);
                }
            }
            // data dropped here
        }
        self.guard.take().expect("guard present")
    }
}

/// RAII guard which holds a mapped read view to a staging buffer. On Drop
/// the buffer is unmapped and returned to the pool.
pub struct StagingMappedReadGuard {
    guard: Option<StagingGuard>,
    // Owned copy of the mapped bytes. We copy contents at map time and
    // immediately unmap the buffer to avoid having live BufferView
    // objects that can conflict with subsequent unmap() calls.
    mapped_data: Option<Vec<u8>>,
}

impl StagingMappedReadGuard {
    /// Access the mapped range as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        self.mapped_data.as_deref().expect("mapped range present")
    }

    /// Take the inner guard if the caller wants to manage the entry manually.
    pub fn take_guard(mut self) -> StagingGuard {
        // mapped bytes are already copied and buffer unmapped at map time
        self.guard.take().expect("guard present")
    }
}

impl Drop for StagingMappedReadGuard {
    fn drop(&mut self) {
        // mapped_data already owns the bytes and the device buffer was
        // unmapped at map time; just drop the guard so the entry returns
        // to the pool.
        self.mapped_data.take();
        self.guard.take();
    }
}

impl Drop for StagingMappedWriteGuard {
    fn drop(&mut self) {
        // If we have an owned mapped Vec, write it back into the staging
        // buffer via the queue before returning the entry to the pool.
        if let Some(data) = self.mapped.take() {
            if let Some(g) = self.guard.as_ref() {
                if let Some(dev) = self.device.as_ref() {
                    dev.queue.write_buffer(g.buffer(), 0, &data);
                    // Poll and wait to ensure the queued write completes so
                    // callers that immediately copy from the staging buffer
                    // see the expected contents.
                    dev.device.poll(wgpu::Maintain::Wait);
                }
            }
        }
        // Drop the guard to return entry to pool
        self.guard.take();
    }
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if let Some(e) = self.entry.take() {
            // mark not in-use and return to pool
            e.in_use.store(false, Ordering::SeqCst);
            return_entry_to_pool(&self.pool, self.key, e);
        }
    }
}

/// Helper to return an entry to the staging pool and enforce per-bucket cap.
fn return_entry_to_pool(pool: &StagingPool, key: NonZeroU64, e: StagingEntry) {
    if let Ok(mut pool) = pool.lock() {
        let deq = pool.entry(key).or_default();
        deq.push_back(e);
        while deq.len() > MAX_PER_BUCKET {
            deq.pop_front();
        }
    }
}

// Minimal wgpu-based GPU backend. This file is intentionally small: it
// demonstrates creating device buffers, uploading host data, dispatching a
// compute shader (assumed to be precompiled SPIR-V), and reading back results.

pub struct GpuDevice {
    // Keep wgpu state minimal for illustration.
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    // size-bucketed staging pool (keyed by bucket size in bytes). Each bucket
    // is an LRU queue (VecDeque) with a small cap to avoid unbounded growth.
    // Stored in an Arc so RAII guards can return entries on Drop without
    // holding a borrow on `self`.
    pub staging_pool:
        std::sync::Arc<std::sync::Mutex<BTreeMap<NonZeroU64, VecDeque<StagingEntry>>>>,
}

/// A staging pool entry: a buffer and an in-use flag so we don't reuse a
/// buffer while it's still being mapped/written by another task.
pub struct StagingEntry {
    pub buffer: wgpu::Buffer,
    pub in_use: AtomicBool,
    pub usage: wgpu::BufferUsages,
}

impl GpuDevice {
    /// Acquire or create a staging entry for `key`. If `initial_contents` is
    /// Some(bytes) and a new buffer is created we initialize it with those
    /// bytes. The returned StagingGuard will return the entry to the pool on
    /// Drop.
    fn acquire_staging_entry(
        &self,
        key: NonZeroU64,
        initial_contents: Option<&[u8]>,
        map_for_write: bool,
    ) -> StagingGuard {
        // try to find a free entry
        if let Ok(mut pool) = self.staging_pool.lock() {
            let deq = pool.entry(key).or_default();
            if let Some(pos) = deq.iter().position(|e| {
                if !e.in_use.load(Ordering::SeqCst) {
                    // ensure the entry's usage matches the requested mapping
                    if map_for_write {
                        // For write-path we will later use the staging buffer as
                        // a copy source into the device-local destination. Make
                        // sure the entry supports COPY_SRC in addition to
                        // COPY_DST so it can be used as the source of copies.
                        return e
                            .usage
                            .contains(wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
                    } else {
                        // For read-path we need MAP_READ so callers can map
                        // the buffer, and COPY_DST so the GPU can copy into it.
                        return e
                            .usage
                            .contains(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
                    }
                }
                false
            }) {
                let e = deq.remove(pos).unwrap();
                e.in_use.store(true, Ordering::SeqCst);
                return StagingGuard {
                    key,
                    entry: Some(e),
                    pool: self.staging_pool.clone(),
                    created_mapped: false,
                };
            }
        }

        // create a new buffer. We must not combine MAP_WRITE with COPY_DST
        // (wgpu validation forbids that). Our write-path avoids mapping the
        // staging buffer at all: callers receive an owned Vec<u8> and we use
        // Queue::write_buffer to upload its contents into the staging buffer
        // when finalized. Therefore write-path buffers get COPY_DST|COPY_SRC
        // usage (no MAP_WRITE). Read-path buffers still use MAP_READ|COPY_DST
        // so they can be mapped for CPU reads after a GPU copy.
        if let Some(bytes) = initial_contents {
            if map_for_write {
                // Create a non-mapped staging buffer and initialize it via
                // queue.write_buffer so we avoid MAP_WRITE usage entirely.
                let usage = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
                let b = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: key.get(),
                    usage,
                    mapped_at_creation: false,
                });
                // Initialize contents using the queue (synchronous wait for prototype)
                self.queue.write_buffer(&b, 0, bytes);
                self.device.poll(wgpu::Maintain::Wait);
                let e = StagingEntry {
                    buffer: b,
                    in_use: AtomicBool::new(true),
                    usage,
                };
                StagingGuard {
                    key,
                    entry: Some(e),
                    pool: self.staging_pool.clone(),
                    created_mapped: false,
                }
            } else {
                // Read-path initial contents: use create_buffer_init as a
                // convenience. The buffer will be created with the provided
                // contents and must allow MAP_READ so callers can map it.
                let usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_SRC;
                let b = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytes,
                        usage,
                    });
                let e = StagingEntry {
                    buffer: b,
                    in_use: AtomicBool::new(true),
                    usage,
                };
                StagingGuard {
                    key,
                    entry: Some(e),
                    pool: self.staging_pool.clone(),
                    created_mapped: true,
                }
            }
        } else {
            if map_for_write {
                // Write-path: staging buffers are not mapped. They must allow
                // being written by the queue (COPY_DST) and later used as
                // copy sources for device-local copies (COPY_SRC).
                let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
                let b = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: key.get(),
                    usage,
                    mapped_at_creation: false,
                });
                let e = StagingEntry {
                    buffer: b,
                    in_use: AtomicBool::new(true),
                    usage,
                };
                StagingGuard {
                    key,
                    entry: Some(e),
                    pool: self.staging_pool.clone(),
                    created_mapped: false,
                }
            } else {
                // Read-path: we need MAP_READ so we can map and copy host
                // visible bytes after a GPU copy. Combine with COPY_DST so
                // GPU can copy into the staging buffer.
                let usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
                let b = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: key.get(),
                    usage,
                    mapped_at_creation: false,
                });
                let e = StagingEntry {
                    buffer: b,
                    in_use: AtomicBool::new(true),
                    usage,
                };
                StagingGuard {
                    key,
                    entry: Some(e),
                    pool: self.staging_pool.clone(),
                    created_mapped: false,
                }
            }
        }
    }

    /// Blocking helper: acquire a staging entry and map it for write. Returns
    /// a StagingMappedWriteGuard which provides mutable access to the mapped
    /// range. This function blocks on map_async completion.
    fn map_staging_write_blocking(
        &self,
        key: NonZeroU64,
        initial_contents: Option<&[u8]>,
    ) -> StagingMappedWriteGuard {
        // Acquire a staging entry; if initial_contents present a mapped-at-creation
        // buffer will be returned already containing the bytes.
        let guard = self.acquire_staging_entry(key, initial_contents, true);
        // For write-path we avoid mapping staging buffers. Present an owned
        // Vec<u8> to the caller: initialize from initial_contents if
        // provided, otherwise zero-filled to the bucket size.
        let mapped_vec = if let Some(init) = initial_contents {
            Some(init.to_vec())
        } else {
            Some(vec![0u8; guard.key.get() as usize])
        };

        StagingMappedWriteGuard {
            guard: Some(guard),
            mapped: mapped_vec,
            device: Some(DeviceManager::global().get_or_init()),
        }
    }

    /// Blocking helper: map an already-acquired staging guard for read and
    /// return a StagingMappedReadGuard. This unmaps and returns the entry
    /// to the pool when the returned guard is dropped.
    fn map_staging_read_blocking_from_guard(&self, guard: StagingGuard) -> StagingMappedReadGuard {
        let mapped_data = if !guard.created_mapped() {
            let buf = guard.buffer();
            let slice = buf.slice(..);
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).ok();
            });
            self.device.poll(wgpu::Maintain::Wait);
            let _ = futures_executor::block_on(rx.receive());
            let v = slice.get_mapped_range();
            // Copy out the mapped bytes into owned Vec<u8>
            let copy = v.to_vec();
            drop(v);
            // Unmap the buffer immediately to avoid live BufferView objects
            buf.unmap();
            Some(copy)
        } else {
            // If the buffer was created with initial contents we don't have
            // a mapped view; fall back to an empty vector sized to the
            // bucket (guard.key) so callers can still access a slice.
            Some(vec![0u8; guard.key.get() as usize])
        };
        StagingMappedReadGuard {
            guard: Some(guard),
            mapped_data,
        }
    }

    /// Async helper for tokio: acquire and map a staging buffer for write,
    /// awaiting map readiness in an async-friendly loop while polling device.
    #[cfg(feature = "tokio")]
    async fn map_staging_write_async(
        &self,
        key: NonZeroU64,
        initial_contents: Option<&[u8]>,
    ) -> StagingMappedWriteGuard {
        // no extra imports needed here; use tokio::time::sleep inline when required
        let guard = self.acquire_staging_entry(key, initial_contents, true);
        // Async write-path: present an owned Vec<u8> without mapping the
        // staging buffer. Initialize from initial_contents or zero-fill.
        let mapped_vec = if let Some(init) = initial_contents {
            Some(init.to_vec())
        } else {
            Some(vec![0u8; guard.key.get() as usize])
        };

        StagingMappedWriteGuard {
            guard: Some(guard),
            mapped: mapped_vec,
            device: Some(DeviceManager::global().get_or_init()),
        }
    }

    /// Async helper: map an already-acquired staging guard for read.
    #[cfg(feature = "tokio")]
    async fn map_staging_read_async_from_guard(
        &self,
        guard: StagingGuard,
    ) -> StagingMappedReadGuard {
        let mapped_data = if !guard.created_mapped() {
            let buf = guard.buffer();
            let slice = buf.slice(..);
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).ok();
            });
            loop {
                tokio::select! {
                    res = rx.receive() => {
                        match res {
                            Some(Ok(())) => break,
                            Some(Err(_)) => panic!("map_async failed"),
                            None => panic!("map_async channel closed"),
                        }
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_millis(1)) => {
                        self.device.poll(wgpu::Maintain::Poll);
                    }
                }
            }
            let v = slice.get_mapped_range();
            let copy = v.to_vec();
            drop(v);
            buf.unmap();
            Some(copy)
        } else {
            Some(vec![0u8; guard.key.get() as usize])
        };
        StagingMappedReadGuard {
            guard: Some(guard),
            mapped_data,
        }
    }

    /// Convenience blocking helper: acquire a staging entry for `key` and
    /// map it for read, returning a StagingMappedReadGuard. This is the
    /// symmetric counterpart to `map_staging_write_blocking`.
    fn map_staging_read_blocking(&self, key: NonZeroU64) -> StagingMappedReadGuard {
        let guard = self.acquire_staging_entry(key, None::<&[u8]>, false);
        self.map_staging_read_blocking_from_guard(guard)
    }

    /// Convenience async helper: acquire a staging entry and map it for
    /// read. This awaits map readiness in an async-friendly loop.
    #[cfg(feature = "tokio")]
    async fn map_staging_read_async(&self, key: NonZeroU64) -> StagingMappedReadGuard {
        let guard = self.acquire_staging_entry(key, None::<&[u8]>, false);
        self.map_staging_read_async_from_guard(guard).await
    }

    // Convenience: acquire and map for read (async) in one call.
    // Async convenience helper removed: use map_staging_read_async_from_guard
    // directly where needed to avoid unused-dead-code warnings.

    pub fn new_from_adapter(adapter: &wgpu::Adapter) -> anyhow::Result<Self> {
        // Create instance and adapter
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))?;
        Ok(GpuDevice {
            device,
            queue,
            staging_pool: std::sync::Arc::new(std::sync::Mutex::new(BTreeMap::new())),
        })
    }

    /// Create a default GpuDevice by selecting a suitable adapter from the
    /// system and initializing it. This is a convenience used by the
    /// prototype DeviceManager; in production you may want explicit adapter
    /// enumeration and selection policies.
    pub fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
        });
        // Honor EENN_GPU_ADAPTER env var if present. It may be an index (0-based)
        // or a substring of the adapter name/vendor to prefer.
        if let Ok(sel) = env::var("EENN_GPU_ADAPTER") {
            // Try parse as index first then substring match on adapter name
            if let Ok(idx) = sel.parse::<usize>()
                && let Some(adapter) = instance.enumerate_adapters(wgpu::Backends::all()).nth(idx)
            {
                return GpuDevice::new_from_adapter(&adapter);
            }
            for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
                if let Some(info) = adapter.get_info().name.get(..)
                    && info.contains(&sel)
                {
                    return GpuDevice::new_from_adapter(&adapter);
                }
            }
        }

        // No explicit selection: pick the first adapter that satisfies basic
        // compute/storage requirements. This can be improved to consider
        // performance metrics or user policies.
        for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
            if adapter_matches_requirements(&adapter) {
                return GpuDevice::new_from_adapter(&adapter);
            }
        }

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow::anyhow!("no suitable GPU adapter found"))?;
        GpuDevice::new_from_adapter(&adapter)
    }

    pub fn create_buffer_from_f32(&self, data: &[f32], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        // Create a device-local buffer and upload via write_buffer (fast path)
        let size = std::mem::size_of_val(data) as u64;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            // Ensure device buffers created via this helper support being
            // used as copy sources for readback as well as copy dst for
            // uploads. Tests expect roundtrip readback so include COPY_SRC.
            usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bytes = bytemuck::cast_slice(data);
        // For small uploads it's faster and simpler to use queue.write_buffer.
        // For large uploads we fall back to a staging copy.
        if bytes.len() <= 64 * 1024 {
            self.queue.write_buffer(&buf, 0, bytes);
            // optionally poll to ensure completion for prototype
            self.device.poll(wgpu::Maintain::Wait);
            buf
        } else {
            // staging path
            let size = bytes.len() as u64;
            // pick a bucket which is the next power-of-two >= size
            let key = bucket_for_size(size);
            // Acquire and map staging buffer for write (blocking helper)
            let mapped_guard = self.map_staging_write_blocking(key, Some(bytes));
            // If mapped, write into mapped slice then finalize mapping by
            // taking the guard which drops the mapped view and unmaps the
            // buffer before we encode/submit GPU copies.
            if mapped_guard.mapped.is_some() {
                // write into mapped slice
                let mut mg = mapped_guard;
                let slice = mg.as_mut_slice();
                slice.copy_from_slice(bytes);
                // take_guard will drop the mapped view and unmap the buffer
                let guard = mg.take_guard();
                let staging_entry_buf = guard.buffer();
                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                encoder.copy_buffer_to_buffer(staging_entry_buf, 0, &buf, 0, size);
                self.queue.submit(Some(encoder.finish()));
                self.device.poll(wgpu::Maintain::Wait);
                // return staging entry to pool by dropping guard
                drop(guard);
            } else {
                // If the buffer was created mapped-at-creation with initial
                // contents it may not have a mapped view; in that case we can
                // directly use the entry's buffer for the copy.
                let staging_entry_buf =
                    &mapped_guard.guard.as_ref().expect("guard present").buffer();
                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                encoder.copy_buffer_to_buffer(staging_entry_buf, 0, &buf, 0, size);
                self.queue.submit(Some(encoder.finish()));
                self.device.poll(wgpu::Maintain::Wait);
                // mapped_guard drops here and returns entry to pool
            }
            buf
        }
    }

    /// Read a device buffer into host-owned Vec<f32> using a staging buffer map.
    pub fn read_buffer_to_host(
        &self,
        buffer: &wgpu::Buffer,
        elements: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let size = (elements * std::mem::size_of::<f32>()) as u64;
        // pick a bucket for staging of at least `size` bytes
        let key = bucket_for_size(size);
        // Acquire a staging guard for readback and issue copy
        let guard = self.acquire_staging_entry(key, None::<&[u8]>, false);
        let staging_buf = guard.buffer();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, staging_buf, 0, size);
        self.queue.submit(Some(encoder.finish()));

        // Map and read using the RAII read guard helper (blocking)
        let mapped_read = self.map_staging_read_blocking_from_guard(guard);
        // The staging bucket may be larger than the requested `size`.
        // Trim to the exact requested byte range before casting to f32 so
        // callers receive exactly `elements` floats.
        let mapped_slice = mapped_read.as_slice();
        let trimmed = &mapped_slice[..(size as usize)];
        let vec: Vec<f32> = bytemuck::cast_slice(trimmed).to_vec();
        // drop mapped_read -> unmap + return to pool
        drop(mapped_read);
        Ok(vec)
    }
}

// Async readback variant using tokio to avoid blocking the thread. This is
// compiled only when the `tokio` optional feature is enabled and provides a
// true non-blocking path for mapping buffers.
#[cfg(feature = "tokio")]
impl GpuDevice {
    pub async fn read_buffer_to_host_async(
        &self,
        buffer: &wgpu::Buffer,
        elements: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let size = (elements * std::mem::size_of::<f32>()) as u64;
        let key = bucket_for_size(size);
        // Acquire a staging guard for async readback
        let guard = self.acquire_staging_entry(key, None::<&[u8]>, false);
        let staging_buf = guard.buffer();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, staging_buf, 0, size);
        self.queue.submit(Some(encoder.finish()));
        // Use the async helper which awaits mapping and returns a mapped guard
        let mapped_read = self.map_staging_read_async_from_guard(guard).await;
        let mapped_slice = mapped_read.as_slice();
        let trimmed = &mapped_slice[..(size as usize)];
        let vec: Vec<f32> = bytemuck::cast_slice(trimmed).to_vec();
        drop(mapped_read);
        Ok(vec)
    }
}

/// Return true if the adapter appears suitable for compute workloads we need.
fn adapter_matches_requirements(adapter: &wgpu::Adapter) -> bool {
    // Deprecated: prefer adapter_score-based selection. Keep a minimal
    // fallback that returns true if the adapter has basic storage support.
    let features = adapter.features();
    features.contains(wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY)
}

/// Compute a heuristic score for an adapter. Higher is better. The scoring
/// is conservative and additive: discrete GPUs, feature support, and
/// larger limits increase the score. When compiled with the
/// `system_analysis` feature we also consult the system analyzer to boost
/// scores based on system-wide GPU capability signals.
fn adapter_score(adapter: &wgpu::Adapter) -> u32 {
    let info = adapter.get_info();
    let features = adapter.features();
    let limits = adapter.limits();

    let mut score: u32 = 0;

    // Prefer discrete GPUs strongly
    match info.device_type {
        wgpu::DeviceType::DiscreteGpu => score += 60,
        wgpu::DeviceType::IntegratedGpu => score += 30,
        wgpu::DeviceType::VirtualGpu => score += 10,
        _ => score += 5,
    }

    // Penalize software renderers
    if info.name.to_lowercase().contains("swiftshader")
        || info.name.to_lowercase().contains("llvmpipe")
    {
        score = score.saturating_sub(50);
    }

    // Feature bonuses
    if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        score += 5;
    }
    if features.contains(wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY) {
        score += 20;
    }
    if features.contains(wgpu::Features::PUSH_CONSTANTS) {
        score += 5;
    }

    // Limits: favor larger alignments and max bindings
    score += limits.max_storage_buffers_per_shader_stage.min(16) * 2;
    if limits.min_storage_buffer_offset_alignment >= 256 {
        score += 10;
    }

    // Vendor heuristics
    let lname = info.vendor.to_string();
    let vname = info.name.to_lowercase();
    if vname.contains("nvidia") || lname.to_lowercase().contains("nvidia") {
        score += 10;
    }

    // system_analysis integration removed from this module to avoid
    // When available, consult the system analysis probe's cached score
    // to slightly adjust ordering. The probe runs in background and may
    // be zero if not yet ready.
    let sys_score = SYS_ANALYSIS_SCORE.load(Ordering::SeqCst);
    if sys_score > 0 {
        // clamp and add a small bonus so that system analysis can influence
        // adapter ordering without overwhelming hardware heuristics.
        score = score.saturating_add((sys_score as u32).min(20));
    }

    score
}

/// Quick, non-blocking adapter score using adapter info only. This avoids
/// invoking any blocking system probes and is safe to call from tests.
fn adapter_score_quick(adapter: &wgpu::Adapter) -> u32 {
    let info = adapter.get_info();
    let mut score: u32 = 0;
    match info.device_type {
        wgpu::DeviceType::DiscreteGpu => score += 50,
        wgpu::DeviceType::IntegratedGpu => score += 25,
        _ => score += 5,
    }
    let name = info.name.to_lowercase();
    if name.contains("nvidia") {
        score += 10;
    } else if name.contains("amd") || name.contains("radeon") {
        score += 8;
    }
    if name.contains("swiftshader") || name.contains("llvmpipe") {
        score = score.saturating_sub(40);
    }
    score
}

/// Compute a bucket key (next power-of-two) for a requested size in bytes.
fn bucket_for_size(size: u64) -> NonZeroU64 {
    let mut bucket = 1u64;
    while bucket < size {
        bucket <<= 1;
    }
    NonZeroU64::new(bucket).unwrap()
}

/// Async-friendly wrappers (prototype): these return Futures but internally
/// perform blocking operations using `futures_executor::block_on` to keep the
/// API stable while avoiding an async runtime dependency in the prototype.
impl GpuDevice {
    /// Async readback: returns a future that resolves to Vec<f32>.
    pub async fn readback_f32_async(
        &self,
        buffer: &wgpu::Buffer,
        elements: usize,
    ) -> anyhow::Result<Vec<f32>> {
        self.read_buffer_to_host(buffer, elements)
    }
}

// Tokio-enabled non-blocking upload path. This attempts to use mapped_at_creation
// when creating a new staging buffer (fast, immediate mapping) and falls back
// to map_async for reused staging buffers. Staging buffers are taken from the
// per-bucket VecDeque pool (popped while in-use) and returned after submission.
#[cfg(feature = "tokio")]
impl GpuDevice {
    pub async fn upload_array_f32_async(
        &self,
        data: Arc<ArrayD<f32>>,
        usage: wgpu::BufferUsages,
    ) -> anyhow::Result<Arc<GpuBuffer>> {
        // use tokio::time::sleep inline when needed to avoid unused import warnings

        let flat: Vec<f32> = if let Some(slice) = data.as_slice_memory_order() {
            slice.to_vec()
        } else {
            // avoid intermediate iterator allocations by preallocating
            let mut v = Vec::with_capacity(data.len());
            v.extend(data.iter().cloned());
            v
        };
        let bytes = bytemuck::cast_slice(&flat);
        let size = bytes.len() as u64;

        // Fast path: small uploads can use write_buffer
        if bytes.len() <= 64 * 1024 {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                // small-path device buffer should also allow readback (COPY_SRC)
                // so include COPY_SRC alongside COPY_DST.
                usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, bytes);
            self.device.poll(wgpu::Maintain::Wait);
            let device_arc = DeviceManager::global().get_or_init();
            let gb = GpuBuffer {
                device: device_arc.clone(),
                buffer: Arc::new(buf),
                dtype: DType::F32,
                shape: data.shape().to_vec(),
            };
            return Ok(Arc::new(gb));
        }

        // staging path: pick a bucket which is the next power-of-two >= size
        let key = bucket_for_size(size);

        // Acquire and map staging buffer for write (async helper)
        let mapped_guard = self.map_staging_write_async(key, Some(bytes)).await;
        if mapped_guard.mapped.is_some() {
            // write into mapped slice then finalize mapping before submitting
            let mut mg = mapped_guard;
            mg.as_mut_slice().copy_from_slice(bytes);
            // take_guard will drop the mapped view and unmap the buffer
            let guard = mg.take_guard();
            // Sanity: ensure the staging entry supports being used as a copy
            // source. If this invariant fails it indicates a reuse bug.
            let usage = guard.entry.as_ref().unwrap().usage;
            assert!(
                usage.contains(wgpu::BufferUsages::COPY_SRC),
                "staging entry missing COPY_SRC usage: {:?}",
                usage
            );
            let staging_buf = guard.buffer();
            // Create destination device-local buffer and encode copy.
            let dst = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                // Destination device-local buffer created for uploads should
                // support being copied-from for later readback in tests.
                usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(staging_buf, 0, &dst, 0, size);
            self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
            // return staging entry to pool
            drop(guard);
            // return dst buffer as device buffer
            let device_arc = DeviceManager::global().get_or_init();
            let gb = GpuBuffer {
                device: device_arc.clone(),
                buffer: Arc::new(dst),
                dtype: DType::F32,
                shape: data.shape().to_vec(),
            };
            return Ok(Arc::new(gb));
        } else {
            // If created mapped-at-creation without an active mapped view,
            // use the buffer directly for the copy and return the created
            // device buffer as a GpuBuffer.
            let guard_ref = mapped_guard.guard.as_ref().expect("guard present");
            let usage = guard_ref.entry.as_ref().unwrap().usage;
            assert!(
                usage.contains(wgpu::BufferUsages::COPY_SRC),
                "staging entry missing COPY_SRC usage: {:?}",
                usage
            );
            let staging_buf = &guard_ref.buffer();
            let dst = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(staging_buf, 0, &dst, 0, size);
            self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
            // mapped_guard drops here and returns entry to pool
            let device_arc = DeviceManager::global().get_or_init();
            let gb = GpuBuffer {
                device: device_arc.clone(),
                buffer: Arc::new(dst),
                dtype: DType::F32,
                shape: data.shape().to_vec(),
            };
            return Ok(Arc::new(gb));
        }
    }
}

// Fallback async uploader when tokio feature is not enabled. This keeps the
// previous behavior (async fn but does work synchronously internally).
#[cfg(not(feature = "tokio"))]
impl GpuDevice {
    /// Async upload: returns a future that resolves to a GpuBuffer boxed in an
    /// Arc. Currently the future executes synchronously internally.
    pub async fn upload_array_f32_async(
        &self,
        data: Arc<ArrayD<f32>>,
        usage: wgpu::BufferUsages,
    ) -> anyhow::Result<Arc<GpuBuffer>> {
        // synchronous work in async wrapper; use the global manager to obtain
        // an Arc<GpuDevice> for the buffer handle so we don't attempt to clone
        // the underlying wgpu device/queue handles.
        let device_arc = DeviceManager::global().get_or_init();
        let flat: Vec<f32> = if let Some(slice) = data.as_slice_memory_order() {
            slice.to_vec()
        } else {
            let mut v = Vec::with_capacity(data.len());
            v.extend(data.iter().cloned());
            v
        };
        let buf = device_arc.create_buffer_from_f32(
            &flat,
            usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        );
        let gb = GpuBuffer {
            device: device_arc.clone(),
            buffer: Arc::new(buf),
            dtype: DType::F32,
            shape: data.shape().to_vec(),
        };
        Ok(Arc::new(gb))
    }
}

// Note: we intentionally do not implement Clone for GpuDevice because
// wgpu::Device/Queue types are not trivially cloneable; sharing is done
// via Arc<GpuDevice> managed by DeviceManager.

// Simple global device pool for the prototype. This uses once_cell Lazy to
// initialize a single GpuDevice once and reuse it for subsequent buffer
// creations. In production you'd want finer-grained control and proper
// device/context lifecycle management.
use once_cell::sync::Lazy;

/// DeviceManager handles one or more GpuDevice instances. For now it keeps a
/// lazily-initialized global device behind a mutex; later this can be
/// extended to manage multiple adapters/devices and more sophisticated
/// scheduling.
struct DeviceManagerInner {
    devices: Vec<Arc<GpuDevice>>,
    rr: usize,
}

pub struct DeviceManager {
    inner: Mutex<DeviceManagerInner>,
}

/// Policy for selecting a device from the DeviceManager pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceSelectionPolicy {
    /// Round-robin selection (default behavior).
    RoundRobin,
    /// Pick the highest-scoring device (best heuristics).
    BestScore,
    /// Pick by index into the initialized device list (modulo length).
    Index(usize),
}

impl DeviceManager {
    fn new() -> Self {
        Self {
            inner: Mutex::new(DeviceManagerInner {
                devices: Vec::new(),
                rr: 0,
            }),
        }
    }

    pub fn global() -> &'static DeviceManager {
        static INSTANCE: Lazy<DeviceManager> = Lazy::new(DeviceManager::new);
        &INSTANCE
    }

    /// Enumerate adapter infos available on the system. This does not create
    /// any Device objects and is safe to call in tests where initializing a
    /// GPU device may hang or be slow.
    pub fn enumerate_adapters_info(&self) -> Vec<wgpu::AdapterInfo> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
        });
        instance
            .enumerate_adapters(wgpu::Backends::all())
            .map(|a| a.get_info())
            .collect()
    }

    /// Choose an adapter index based on a selection policy without creating
    /// a GpuDevice. Returns None if no adapters are found.
    pub fn select_adapter_index_by_policy(&self, policy: DeviceSelectionPolicy) -> Option<usize> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
        });
        let adapters: Vec<wgpu::Adapter> =
            instance.enumerate_adapters(wgpu::Backends::all()).collect();
        if adapters.is_empty() {
            return None;
        }
        match policy {
            DeviceSelectionPolicy::RoundRobin => Some(0),
            DeviceSelectionPolicy::Index(i) => Some(i % adapters.len()),
            DeviceSelectionPolicy::BestScore => {
                // Score adapters by a quick heuristic (non-blocking). We avoid
                // calling potentially-blocking system-analysis probes here so
                // this function remains safe to call in tests.
                let mut scored: Vec<(u32, usize)> = adapters
                    .iter()
                    .enumerate()
                    .map(|(idx, a)| (adapter_score_quick(a), idx))
                    .collect();
                scored.sort_by(|a, b| b.0.cmp(&a.0));
                Some(scored.first().map(|t| t.1).unwrap_or(0))
            }
        }
    }

    /// Initialize by enumerating adapters and creating a GpuDevice for each
    /// adapter that can be initialized. If no adapter can be initialized we
    /// fall back to the single-adapter `GpuDevice::new()` behavior.
    fn ensure_initialized_locked(&self, guard: &mut std::sync::MutexGuard<'_, DeviceManagerInner>) {
        if !guard.devices.is_empty() {
            return;
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
        });
        // enumerate adapters and score them; prefer higher-scoring adapters.
        let mut scored: Vec<(u32, wgpu::Adapter)> = instance
            .enumerate_adapters(wgpu::Backends::all())
            .map(|a| (adapter_score(&a), a))
            .collect();
        // allow env-based selection (index or substring) to override ordering
        if let Ok(sel) = env::var("EENN_GPU_ADAPTER") {
            if let Ok(idx) = sel.parse::<usize>() {
                if let Some((_, adapter)) = scored.get(idx) {
                    match GpuDevice::new_from_adapter(adapter) {
                        Ok(dev) => guard.devices.push(Arc::new(dev)),
                        Err(e) => eprintln!("warning: failed to init adapter: {}", e),
                    }
                }
            } else {
                // substring match name
                for (_, adapter) in &scored {
                    if adapter
                        .get_info()
                        .name
                        .to_lowercase()
                        .contains(&sel.to_lowercase())
                    {
                        match GpuDevice::new_from_adapter(adapter) {
                            Ok(dev) => guard.devices.push(Arc::new(dev)),
                            Err(e) => eprintln!("warning: failed to init adapter: {}", e),
                        }
                        break;
                    }
                }
            }
        }

        // Sort descending by score and initialize the highest-scoring
        // adapter first. Creating devices for every adapter can be slow or
        // hang on some platforms/drivers; prefer a fast-path that picks the
        // best adapter and initializes it. If you want multi-device
        // initialization, we can make that opt-in later.
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        for (_score, adapter) in scored {
            match GpuDevice::new_from_adapter(&adapter) {
                Ok(dev) => {
                    guard.devices.push(Arc::new(dev));
                    // Stop after first successful initialization to avoid
                    // long stalls during test runs.
                    break;
                }
                Err(e) => {
                    eprintln!("warning: failed to init adapter: {}", e);
                }
            }
        }
        if guard.devices.is_empty() {
            // fallback: try the request_adapter path (block on the future)
            let maybe_adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }));
            if let Some(adapter) = maybe_adapter {
                match GpuDevice::new_from_adapter(&adapter) {
                    Ok(dev) => guard.devices.push(Arc::new(dev)),
                    Err(e) => panic!("failed to init any gpu device: {}", e),
                }
            } else {
                panic!("no gpu adapters found");
            }
        }
    }

    /// Get a device using round-robin selection.
    pub fn get_device_round_robin(&self) -> Arc<GpuDevice> {
        let mut guard = self.inner.lock().unwrap();
        self.ensure_initialized_locked(&mut guard);
        let idx = guard.rr % guard.devices.len();
        guard.rr = guard.rr.wrapping_add(1);
        guard.devices[idx].clone()
    }

    /// Get a device according to a selection policy.
    pub fn get_device_by_policy(&self, policy: DeviceSelectionPolicy) -> Arc<GpuDevice> {
        let mut guard = self.inner.lock().unwrap();
        self.ensure_initialized_locked(&mut guard);
        match policy {
            DeviceSelectionPolicy::RoundRobin => self.get_device_round_robin(),
            DeviceSelectionPolicy::BestScore => {
                // devices are initialized in score-descending order; pick first
                guard
                    .devices
                    .first()
                    .cloned()
                    .expect("no devices available")
            }
            DeviceSelectionPolicy::Index(i) => {
                if guard.devices.is_empty() {
                    panic!("no devices available");
                }
                let idx = i % guard.devices.len();
                guard.devices[idx].clone()
            }
        }
    }

    /// Convenience: get a device (round-robin by default).
    pub fn get_or_init(&self) -> Arc<GpuDevice> {
        self.get_device_round_robin()
    }
}

pub struct GpuBuffer {
    pub device: Arc<GpuDevice>,
    pub buffer: Arc<wgpu::Buffer>,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl DeviceBuffer for GpuBuffer {
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
    fn to_host_f32(&self) -> anyhow::Result<Vec<f32>> {
        // Map buffer to host and copy (simple synchronous approach)
        let size = self.shape.iter().product::<usize>() * std::mem::size_of::<f32>();
        let buffer: &wgpu::Buffer = self.buffer.as_ref();
        // Create a staging buffer
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size as u64);
        self.device.queue.submit(Some(encoder.finish()));
        let buffer_slice = staging.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.device.poll(wgpu::Maintain::Wait);
        let _ = futures_executor::block_on(rx.receive());
        let data = buffer_slice.get_mapped_range();
        let vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(vec)
    }
    fn box_clone(&self) -> Box<dyn DeviceBuffer> {
        Box::new(GpuBuffer {
            device: self.device.clone(),
            buffer: self.buffer.clone(),
            dtype: self.dtype,
            shape: self.shape.clone(),
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl GpuBuffer {
    /// Access the underlying wgpu::Buffer. This is a convenience for host
    /// code that needs to create bind groups or dispatch compute using the
    /// raw buffer handle.
    pub fn raw_buffer(&self) -> &wgpu::Buffer {
        self.buffer.as_ref()
    }
}

/// Convenience: create a boxed DeviceBuffer from an ArrayD<f32> by
/// creating a fresh device and buffer. This is a simple helper for the
/// prototype; in a real system you'd manage a device pool.
pub fn buffer_from_array(a: Arc<ArrayD<f32>>) -> anyhow::Result<Box<dyn DeviceBuffer>> {
    // Get or initialize a device from the manager
    let device = DeviceManager::global().get_or_init();
    let flat: Vec<f32> = if let Some(slice) = a.as_slice_memory_order() {
        slice.to_vec()
    } else {
        let mut v = Vec::with_capacity(a.len());
        v.extend(a.iter().cloned());
        v
    };
    let buf = device.create_buffer_from_f32(
        &flat,
        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    );
    let gb = GpuBuffer {
        device: device.clone(),
        buffer: Arc::new(buf),
        dtype: DType::F32,
        shape: a.shape().to_vec(),
    };
    Ok(Box::new(gb))
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::*;

    #[test]
    fn smoke_selection_api() {
        // Ensure the DeviceManager selection helpers work without creating
        // actual GpuDevice instances (some drivers/platforms may hang on
        // device initialization; this test avoids instantiating devices).
        let manager = DeviceManager::global();
        let infos = manager.enumerate_adapters_info();
        // At least check we can enumerate adapters info (may be empty on CI)
        let _ = infos.len();
        // Selection by index/policy without initializing devices
        let _idx0 = manager.select_adapter_index_by_policy(DeviceSelectionPolicy::Index(0));
        let _best = manager.select_adapter_index_by_policy(DeviceSelectionPolicy::BestScore);
        let _rr = manager.select_adapter_index_by_policy(DeviceSelectionPolicy::RoundRobin);
    }

    #[test]
    fn staging_mapped_read_guard_lifecycle() {
        let device = DeviceManager::global().get_or_init();
        // pick a small bucket
        let size = 1024u64;
        let mut bucket = 1u64;
        while bucket < size {
            bucket <<= 1;
        }
        let key = NonZeroU64::new(bucket).unwrap();

        // acquire an entry and map for read using the blocking helper
        let guard = device.acquire_staging_entry(key, None::<&[u8]>, false);
        let mapped = device.map_staging_read_blocking_from_guard(guard);
        // mapped view should have at least bucket bytes
        assert_eq!(mapped.as_slice().len() as u64, bucket);
        // dropping mapped should unmap and return to pool
        drop(mapped);

        // ensure pool contains an entry for the key which is not in use
        if let Ok(pool) = device.staging_pool.lock() {
            if let Some(deq) = pool.get(&key) {
                assert!(deq.iter().any(|e| !e.in_use.load(Ordering::SeqCst)));
            } else {
                panic!("expected pool bucket present");
            }
        } else {
            panic!("failed to lock staging_pool");
        }
    }

    #[test]
    fn staging_mapped_write_guard_lifecycle() {
        let device = DeviceManager::global().get_or_init();
        // pick a small bucket
        let size = 1024u64;
        let key = bucket_for_size(size);

        // acquire and map for write using the blocking helper
        let mut mapped = device.map_staging_write_blocking(key, None::<&[u8]>);
        if mapped.mapped.as_mut().is_some() {
            // write a simple pattern
            let slice = mapped.as_mut_slice();
            for (i, b) in slice.iter_mut().enumerate().take(16) {
                *b = (i & 0xff) as u8;
            }
        }
        // drop mapped -> unmap + return to pool
        drop(mapped);

        // ensure pool contains an entry for the key which is not in use
        if let Ok(pool) = device.staging_pool.lock() {
            if let Some(deq) = pool.get(&key) {
                assert!(deq.iter().any(|e| !e.in_use.load(Ordering::SeqCst)));
            } else {
                panic!("expected pool bucket present");
            }
        } else {
            panic!("failed to lock staging_pool");
        }
    }
}

// Tokio-enabled GPU integration tests. These run only when both `gpu` and
// `tokio` features are enabled for the crate and a real adapter is available.
#[cfg(all(test, feature = "gpu", feature = "tokio"))]
mod gpu_tokio_tests {
    use super::*;
    use ndarray::ArrayD;

    #[tokio::test]
    async fn async_small_upload_roundtrip() -> anyhow::Result<()> {
        let device = DeviceManager::global().get_or_init();
        // small array (<64KB)
        let a = Arc::new(ArrayD::from_elem(vec![4usize], std::f32::consts::PI));
        let gb = device
            .upload_array_f32_async(a.clone(), wgpu::BufferUsages::STORAGE)
            .await?;
        let elements = a.len();
        let host = device
            .read_buffer_to_host_async(gb.raw_buffer(), elements)
            .await?;
        assert_eq!(host.len(), elements);
        Ok(())
    }

    #[tokio::test]
    async fn async_large_upload_roundtrip() -> anyhow::Result<()> {
        let device = DeviceManager::global().get_or_init();
        // large array (>64KB). 20000 floats -> 80KB
        let mut vec = vec![0.0f32; 20000];
        fill_arange_mul(&mut vec, 0.125f32);
        let a = Arc::new(ArrayD::from_shape_vec(vec![20000usize], vec.clone()).unwrap());
        let gb = device
            .upload_array_f32_async(a.clone(), wgpu::BufferUsages::STORAGE)
            .await?;
        let host = device
            .read_buffer_to_host_async(gb.raw_buffer(), a.len())
            .await?;
        assert_eq!(host.len(), a.len());
        // spot-check a few values
        assert_eq!(host[0], vec[0]);
        assert_eq!(host[1234], vec[1234]);
        assert_eq!(host[19999], vec[19999]);
        Ok(())
    }

    #[tokio::test]
    async fn async_strict_exact_roundtrip() -> anyhow::Result<()> {
        let device = DeviceManager::global().get_or_init();
        // Choose a size comfortably > 64KB to exercise staging buckets.
        let num_f32 = 30000usize; // 120KB
        let size_bytes = (num_f32 * std::mem::size_of::<f32>()) as u64;
        let key = bucket_for_size(size_bytes);

        // Acquire and map staging buffer for write (async helper)
        let mut mapped = device.map_staging_write_async(key, None::<&[u8]>).await;
        // Fill the full mapped range with a deterministic byte pattern
        if let Some(_) = mapped.mapped.as_ref() {
            let slice = mapped.as_mut_slice();
            for i in 0..slice.len() {
                slice[i] = (i & 0xFF) as u8;
            }
        }

        // Finalize the mapped write by taking the guard so the owned bytes
        // are written back into the staging buffer before we perform the
        // GPU copy. This ensures the copy reads the expected data.
        let guard = mapped.take_guard();

        // Create destination device-local buffer and copy staging -> dst
        let dst = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = guard.buffer();
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(staging_buf, 0, &dst, 0, size_bytes);
        device.queue.submit(Some(encoder.finish()));
        device.device.poll(wgpu::Maintain::Wait);

        // return staging entry to pool by dropping the guard
        drop(guard);

        // Read back from dst using async helper (via read_buffer_to_host_async)
        let host = device.read_buffer_to_host_async(&dst, num_f32).await?;
        // host is Vec<f32> constructed from the bytes; reconstruct original bytes
        let mut expected_bytes = Vec::with_capacity((num_f32 * 4) as usize);
        for i in 0..(num_f32 * 4) {
            expected_bytes.push((i & 0xFF) as u8);
        }
        let host_bytes = bytemuck::cast_slice::<f32, u8>(&host);
        if host_bytes != expected_bytes.as_slice() {
            eprintln!(
                "mismatch: host len={} expected len={}",
                host_bytes.len(),
                expected_bytes.len()
            );
            eprintln!(
                "host first 64: {:?}",
                &host_bytes[..host_bytes.len().min(64)]
            );
            eprintln!(
                "exp  first 64: {:?}",
                &expected_bytes[..expected_bytes.len().min(64)]
            );
        }
        assert_eq!(host_bytes, expected_bytes.as_slice());
        Ok(())
    }

    #[tokio::test]
    async fn staging_pool_reuse_and_cap() -> anyhow::Result<()> {
        let device = DeviceManager::global().get_or_init();
        // create multiple large uploads to the same bucket and ensure pool cap
        let mut handles = Vec::new();
        for _ in 0..12 {
            let mut vec = vec![0.0f32; 20000];
            fill_arange_mul(&mut vec, 1.0f32);
            let a = Arc::new(ArrayD::from_shape_vec(vec![20000usize], vec.clone()).unwrap());
            let h = device.upload_array_f32_async(a.clone(), wgpu::BufferUsages::STORAGE);
            handles.push(h);
        }
        // await all
        for h in handles {
            let _ = h.await?;
        }
        // check pool size for the bucket
        let size = (20000 * std::mem::size_of::<f32>()) as u64;
        let key = bucket_for_size(size);
        if let Ok(pool) = device.staging_pool.lock()
            && let Some(deq) = pool.get(&key)
        {
            assert!(deq.len() <= 4, "pool exceeded cap");
        }
        Ok(())
    }
}

impl GpuDevice {
    /// Load a WGSL module and dispatch a compute shader that reads from
    /// `src` and writes to `dst`. This helper assumes the shader has two
    /// storage buffer bindings at set=0 binding=0 and binding=1.
    /// `elements` is the number of f32 elements to process.
    ///
    /// Note: for a rust-gpu/SPIR-V integration we'd add a separate path that
    /// accepts SPIR-V words and uses the `spirv` feature in `wgpu`.
    pub fn dispatch_wgsl_copy(
        &self,
        wgsl_source: &str,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        elements: u32,
    ) -> Result<()> {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl_source)),
            });

        let bind_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "main",
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_x = elements.div_ceil(64);
            cpass.dispatch_workgroups(wg_x, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        // Wait for completion synchronously for the prototype.
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    /// Dispatch a SPIR-V module (given as u32 words) that copies from src to dst.
    pub fn dispatch_spirv_copy(
        &self,
        spirv_words: &[u32],
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        elements: u32,
    ) -> Result<()> {
        // wgpu util helper make_spirv expects bytes; we can use its API if
        // the `spirv` feature is enabled on wgpu. Convert u32 words to &[u8].
        let bytes: &[u8] = bytemuck::cast_slice(spirv_words);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::util::make_spirv(bytes),
            });

        let bind_layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "main",
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_x = elements.div_ceil(64);
            cpass.dispatch_workgroups(wg_x, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }
}
