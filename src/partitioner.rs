use std::cell::RefCell;
use std::sync::Arc;
use wide::f32x4;

// Thread-local scratch area to avoid per-call allocations inside partition().
// Using thread-local storage keeps it safe for concurrent runs on different
// threads while reusing buffers across repeated calls on the same thread.
thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::new());
}

struct Scratch {
    pref_compute: Vec<f64>,
    pref_transfer: Vec<usize>,
    a: Vec<f32>,
    prev: Vec<f32>,
    cur: Vec<f32>,
    best_from: Vec<f32>,
    dp_lminus1: Vec<f32>,
    val: Vec<f32>,
    // Temporary per-block maxima scratch to avoid allocating Vec in hot DP loop
    block_max: Vec<f32>,
    // Instrumentation counters
    instr_total_calls: u64,
    instr_greedy_calls: u64,
    instr_dp_calls: u64,
    // Histogram of dp window_len observed (index = window_len, value = count)
    instr_dp_window_hist: Vec<u64>,
    // Total nanoseconds spent executing DP for a given window size (index = window_len)
    instr_dp_time_ns: Vec<u128>,
    // Per-phase accumulated nanoseconds for DP components (index = window_len)
    instr_dp_time_a_ns: Vec<u128>,
    instr_dp_time_best_from_ns: Vec<u128>,
    instr_dp_time_cur_ns: Vec<u128>,
    instr_dp_time_rotate_ns: Vec<u128>,
    instr_dp_time_select_ns: Vec<u128>,
    // Cycle counters for DP and phases (index = window_len)
    instr_dp_cycles: Vec<u128>,
    instr_dp_cycles_a: Vec<u128>,
    instr_dp_cycles_best_from: Vec<u128>,
    instr_dp_cycles_cur: Vec<u128>,
    instr_dp_cycles_rotate: Vec<u128>,
    instr_dp_cycles_select: Vec<u128>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            pref_compute: Vec::new(),
            pref_transfer: Vec::new(),
            a: Vec::new(),
            prev: Vec::new(),
            cur: Vec::new(),
            best_from: Vec::new(),
            dp_lminus1: Vec::new(),
            val: Vec::new(),
            instr_total_calls: 0,
            instr_greedy_calls: 0,
            instr_dp_calls: 0,
            instr_dp_window_hist: Vec::new(),
            instr_dp_time_ns: Vec::new(),
            instr_dp_time_a_ns: Vec::new(),
            instr_dp_time_best_from_ns: Vec::new(),
            instr_dp_time_cur_ns: Vec::new(),
            instr_dp_time_rotate_ns: Vec::new(),
            instr_dp_time_select_ns: Vec::new(),
            instr_dp_cycles: Vec::new(),
            instr_dp_cycles_a: Vec::new(),
            instr_dp_cycles_best_from: Vec::new(),
            instr_dp_cycles_cur: Vec::new(),
            instr_dp_cycles_rotate: Vec::new(),
            instr_dp_cycles_select: Vec::new(),
            block_max: Vec::new(),
        }
    }

    /// Ensure internal buffers are at least `len` long and zeroed.
    fn ensure_len(&mut self, len: usize) {
        if self.pref_compute.len() < len {
            self.pref_compute.resize(len, 0.0);
        }
        if self.pref_transfer.len() < len {
            self.pref_transfer.resize(len, 0);
        }
        if self.a.len() < len {
            self.a.resize(len, 0.0_f32);
        }
        if self.prev.len() < len {
            self.prev.resize(len, 0.0_f32);
        }
        if self.cur.len() < len {
            self.cur.resize(len, f32::NEG_INFINITY);
        }
        if self.best_from.len() < len + 1 {
            self.best_from.resize(len + 1, f32::NEG_INFINITY);
        }
        if self.dp_lminus1.len() < len {
            self.dp_lminus1.resize(len, f32::NEG_INFINITY);
        }
        if self.instr_dp_window_hist.len() < len {
            self.instr_dp_window_hist.resize(len, 0);
        }
        if self.instr_dp_time_ns.len() < len {
            self.instr_dp_time_ns.resize(len, 0);
        }
        if self.instr_dp_time_a_ns.len() < len {
            self.instr_dp_time_a_ns.resize(len, 0);
        }
        if self.instr_dp_time_best_from_ns.len() < len {
            self.instr_dp_time_best_from_ns.resize(len, 0);
        }
        if self.instr_dp_time_cur_ns.len() < len {
            self.instr_dp_time_cur_ns.resize(len, 0);
        }
        if self.instr_dp_time_rotate_ns.len() < len {
            self.instr_dp_time_rotate_ns.resize(len, 0);
        }
        if self.instr_dp_time_select_ns.len() < len {
            self.instr_dp_time_select_ns.resize(len, 0);
        }
        if self.instr_dp_cycles.len() < len {
            self.instr_dp_cycles.resize(len, 0);
        }
        if self.instr_dp_cycles_a.len() < len {
            self.instr_dp_cycles_a.resize(len, 0);
        }
        if self.instr_dp_cycles_best_from.len() < len {
            self.instr_dp_cycles_best_from.resize(len, 0);
        }
        if self.instr_dp_cycles_cur.len() < len {
            self.instr_dp_cycles_cur.resize(len, 0);
        }
        if self.instr_dp_cycles_rotate.len() < len {
            self.instr_dp_cycles_rotate.resize(len, 0);
        }
        if self.instr_dp_cycles_select.len() < len {
            self.instr_dp_cycles_select.resize(len, 0);
        }
        if self.val.len() < len {
            self.val.resize(len, 0.0_f32);
        }
        // ensure block_max can hold ceil((len+1)/block) blocks for block-size 8
        let block = 8usize;
        let blocks_needed = (len + block) / block + 2; // +2 safety
        if self.block_max.len() < blocks_needed {
            self.block_max.resize(blocks_needed, f32::NEG_INFINITY);
        }
    }
}

/// Dump instrumentation counters to target/criterion/partitioner_instrumentation.json
pub fn dump_instrumentation() {
    use serde::Serialize;
    use std::fs::File;
    use std::fs::create_dir_all;
    use std::io::Write;

    #[derive(Serialize)]
    struct Instr {
        total_calls: u64,
        greedy_calls: u64,
        dp_calls: u64,
        dp_window_hist: Vec<u64>,
        dp_time_ns: Vec<u128>,
        dp_cycles: Vec<u128>,
        dp_time_a_ns: Vec<u128>,
        dp_time_best_from_ns: Vec<u128>,
        dp_time_cur_ns: Vec<u128>,
        dp_time_rotate_ns: Vec<u128>,
        dp_time_select_ns: Vec<u128>,
        dp_cycles_a: Vec<u128>,
        dp_cycles_best_from: Vec<u128>,
        dp_cycles_cur: Vec<u128>,
        dp_cycles_rotate: Vec<u128>,
        dp_cycles_select: Vec<u128>,
    }

    let instr = SCRATCH.with(|cell| {
        let s = cell.borrow();
        Instr {
            total_calls: s.instr_total_calls,
            greedy_calls: s.instr_greedy_calls,
            dp_calls: s.instr_dp_calls,
            dp_window_hist: s.instr_dp_window_hist.clone(),
            dp_time_ns: s.instr_dp_time_ns.clone(),
            dp_cycles: s.instr_dp_cycles.clone(),
            dp_time_a_ns: s.instr_dp_time_a_ns.clone(),
            dp_time_best_from_ns: s.instr_dp_time_best_from_ns.clone(),
            dp_time_cur_ns: s.instr_dp_time_cur_ns.clone(),
            dp_time_rotate_ns: s.instr_dp_time_rotate_ns.clone(),
            dp_time_select_ns: s.instr_dp_time_select_ns.clone(),
            dp_cycles_a: s.instr_dp_cycles_a.clone(),
            dp_cycles_best_from: s.instr_dp_cycles_best_from.clone(),
            dp_cycles_cur: s.instr_dp_cycles_cur.clone(),
            dp_cycles_rotate: s.instr_dp_cycles_rotate.clone(),
            dp_cycles_select: s.instr_dp_cycles_select.clone(),
        }
    });

    let out_dir = std::path::Path::new("target").join("criterion");
    let _ = create_dir_all(&out_dir);
    let path = out_dir.join("partitioner_instrumentation.json");
    if let Ok(mut f) = File::create(&path) {
        if let Ok(s) = serde_json::to_string_pretty(&instr) {
            let _ = f.write_all(s.as_bytes());
        }
    }
}

/// Dump instrumentation with an explicit benchmark id appended to the filename.
pub fn dump_instrumentation_for(id: &str) {
    use serde::Serialize;
    use std::fs::File;
    use std::fs::create_dir_all;
    use std::io::Write;

    #[derive(Serialize)]
    struct Instr {
        total_calls: u64,
        greedy_calls: u64,
        dp_calls: u64,
        dp_window_hist: Vec<u64>,
        dp_time_ns: Vec<u128>,
        dp_cycles: Vec<u128>,
        dp_time_a_ns: Vec<u128>,
        dp_time_best_from_ns: Vec<u128>,
        dp_time_cur_ns: Vec<u128>,
        dp_time_rotate_ns: Vec<u128>,
        dp_time_select_ns: Vec<u128>,
        dp_cycles_a: Vec<u128>,
        dp_cycles_best_from: Vec<u128>,
        dp_cycles_cur: Vec<u128>,
        dp_cycles_rotate: Vec<u128>,
        dp_cycles_select: Vec<u128>,
    }

    let instr = SCRATCH.with(|cell| {
        let s = cell.borrow();
        Instr {
            total_calls: s.instr_total_calls,
            greedy_calls: s.instr_greedy_calls,
            dp_calls: s.instr_dp_calls,
            dp_window_hist: s.instr_dp_window_hist.clone(),
            dp_time_ns: s.instr_dp_time_ns.clone(),
            dp_cycles: s.instr_dp_cycles.clone(),
            dp_time_a_ns: s.instr_dp_time_a_ns.clone(),
            dp_time_best_from_ns: s.instr_dp_time_best_from_ns.clone(),
            dp_time_cur_ns: s.instr_dp_time_cur_ns.clone(),
            dp_time_rotate_ns: s.instr_dp_time_rotate_ns.clone(),
            dp_time_select_ns: s.instr_dp_time_select_ns.clone(),
            dp_cycles_a: s.instr_dp_cycles_a.clone(),
            dp_cycles_best_from: s.instr_dp_cycles_best_from.clone(),
            dp_cycles_cur: s.instr_dp_cycles_cur.clone(),
            dp_cycles_rotate: s.instr_dp_cycles_rotate.clone(),
            dp_cycles_select: s.instr_dp_cycles_select.clone(),
        }
    });

    let out_dir = std::path::Path::new("target").join("criterion");
    let _ = create_dir_all(&out_dir);
    // sanitize id for filename safety: keep ASCII alphanum, underscore, dash
    let safe: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let filename = format!("partitioner_instrumentation_{}.json", safe);
    let path = out_dir.join(filename);
    if let Ok(mut f) = File::create(&path) {
        if let Ok(s) = serde_json::to_string_pretty(&instr) {
            let _ = f.write_all(s.as_bytes());
        }
    }
}

/// A minimal Op trait for partitioner testing. In the full runtime this would be
/// a richer trait with codegen hooks and metadata. Here we just expose whether
/// the op is GPU-capable and a simple compute/transfer estimate.
pub trait Op: Send + Sync {
    fn name(&self) -> &str;
    fn gpu_capable(&self) -> bool;
    /// Rough cost model: compute_ms on GPU, and transfer_bytes to move inputs/outputs.
    fn estimate(&self) -> (f64 /*compute_ms*/, usize /*transfer_bytes*/);
}

/// Segment describes a contiguous run of ops planned to run on GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub start: usize,
    pub end: usize, // exclusive
}

pub trait Partitioner: Send + Sync {
    fn partition(&self, ops: &[Arc<dyn Op>]) -> Vec<Segment>;
}

/// Greedy partitioner: accumulate consecutive gpu-capable ops while benefit > 0.
/// Benefit heuristic: sum(compute_ms) - (transfer_bytes / bytes_per_ms).
pub struct GreedyPartitioner {
    pub lookahead: usize,
    pub bytes_per_ms: f64,
    pub max_ops_in_fusion: usize,
}

impl Default for GreedyPartitioner {
    fn default() -> Self {
        Self {
            lookahead: 2,
            bytes_per_ms: 1_000_000.0, /* placeholder */
            max_ops_in_fusion: 128,
        }
    }
}

impl Partitioner for GreedyPartitioner {
    fn partition(&self, ops: &[Arc<dyn Op>]) -> Vec<Segment> {
        let mut segments = Vec::new();
        let mut i = 0;
        while i < ops.len() {
            if !ops[i].gpu_capable() {
                i += 1;
                continue;
            }

            // Determine contiguous gpu-capable window starting at i (do not cross non-gpu ops)
            let mut contiguous_end = i;
            while contiguous_end < ops.len()
                && contiguous_end < i + self.max_ops_in_fusion
                && ops[contiguous_end].gpu_capable()
            {
                contiguous_end += 1;
            }
            let max_end = contiguous_end;
            let window_len = max_end - i;

            // If no window, skip
            if window_len == 0 {
                i += 1;
                continue;
            }

            // Precompute prefix sums for compute and transfer to evaluate any interval quickly.
            SCRATCH.with(|cell| {
                let mut s = cell.borrow_mut();
                s.instr_total_calls += 1;
                let need = window_len + 1;
                s.ensure_len(need);
                for k in 0..window_len {
                    let (c, t) = ops[i + k].estimate();
                    s.pref_compute[k + 1] = s.pref_compute[k] + c;
                    s.pref_transfer[k + 1] = s.pref_transfer[k] + t;
                }

                if self.lookahead <= 1 {
                    s.instr_greedy_calls += 1;
                    // original greedy incremental growth behavior
                    let mut best_end = i;
                    for rel in 0..window_len {
                        // use precomputed prefix sums to avoid dynamic dispatch in the hot loop
                        let sum_compute = s.pref_compute[rel + 1];
                        let sum_transfer = s.pref_transfer[rel + 1];
                        let benefit = sum_compute - (sum_transfer as f64 / self.bytes_per_ms);
                        if benefit > 0.0 {
                            best_end = i + rel + 1;
                        } else {
                            break;
                        }
                    }
                    if best_end > i {
                        segments.push(Segment {
                            start: i,
                            end: best_end,
                        });
                        i = best_end;
                    } else {
                        i += 1;
                    }
                    return;
                }

                // Dynamic programming scorer: compute best total benefit for up to L segments
                // over the window. We compute dp[k][pos] = best total benefit achievable
                // starting at relative position `pos` using at most k segments. We only
                // need k up to lookahead, and pos in [0..=window_len].

                s.instr_dp_calls += 1;
                // record observed window size
                if window_len < s.instr_dp_window_hist.len() {
                    s.instr_dp_window_hist[window_len] += 1;
                }

                // Time the DP computation for this window size and accumulate (cycles)
                let dp_start = std::time::Instant::now();
                // rdtsc helper defined below (once)

                let cycles_start = rdtsc();
                // rdtsc helper
                #[inline]
                fn rdtsc() -> u128 {
                    // Use platform intrinsics when available
                    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                    unsafe {
                        return core::arch::x86_64::_rdtsc() as u128;
                    }
                    #[cfg(all(target_arch = "x86", target_feature = "sse2"))]
                    unsafe {
                        return core::arch::x86::_rdtsc() as u128;
                    }
                    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                    {
                        0
                    }
                    #[cfg(all(
                        any(target_arch = "x86", target_arch = "x86_64"),
                        not(target_feature = "sse2")
                    ))]
                    {
                        0
                    }
                }

                // Per-phase accumulators for this DP invocation (cycles)
                let mut phase_a_cycles: u128 = 0;
                let mut phase_best_from_cycles: u128 = 0;
                let mut phase_cur_cycles: u128 = 0;
                let mut phase_rotate_cycles: u128 = 0;
                let mut phase_select_cycles: u128 = 0;

                let l = self.lookahead.min(window_len);

                // Rolling DP: keep only previous and current dp rows to save memory.
                s.prev[..=window_len].fill(0.0);
                for v in s.cur.iter_mut() {
                    *v = f32::NEG_INFINITY;
                }

                // Precompute A[x] = pref_compute[x] - pref_transfer[x] / bytes_per_ms
                let c_a0 = rdtsc();
                // Vectorized computation of a[x] = pref_compute[x] - pref_transfer[x] / bytes_per_ms
                let n_a = window_len + 1;
                let mut ax = 0usize;
                let inv_bps = 1.0_f32 / (self.bytes_per_ms as f32);
                while ax + 4 <= n_a {
                    let pc0 = s.pref_compute[ax] as f32;
                    let pc1 = s.pref_compute[ax + 1] as f32;
                    let pc2 = s.pref_compute[ax + 2] as f32;
                    let pc3 = s.pref_compute[ax + 3] as f32;
                    let pt0 = s.pref_transfer[ax] as f32;
                    let pt1 = s.pref_transfer[ax + 1] as f32;
                    let pt2 = s.pref_transfer[ax + 2] as f32;
                    let pt3 = s.pref_transfer[ax + 3] as f32;
                    let pv = f32x4::new([pc0, pc1, pc2, pc3]);
                    let tv = f32x4::new([pt0, pt1, pt2, pt3]) * f32x4::splat(inv_bps);
                    let av = pv - tv;
                    let out = av.to_array();
                    s.a[ax] = out[0];
                    s.a[ax + 1] = out[1];
                    s.a[ax + 2] = out[2];
                    s.a[ax + 3] = out[3];
                    ax += 4;
                }
                while ax < n_a {
                    s.a[ax] = (s.pref_compute[ax] as f32) - (s.pref_transfer[ax] as f32 * inv_bps);
                    ax += 1;
                }
                let c_a1 = rdtsc();
                phase_a_cycles += c_a1.wrapping_sub(c_a0);

                // We'll capture dp[l-1] when needed for final selection
                let mut captured = false;
                for k in 1..=l {
                    // best_from[e] = max_{u >= e} (prev[u] + a[u])
                    // Merge computation of best_from (reverse scan) and cur (uses best_from[e+1])
                    let c_b0 = rdtsc();
                    // forward pass: compute val[e] = prev[e] + a[e]
                    // Vectorized forward pass: val[e] = prev[e] + a[e]
                    let n = window_len + 1;
                    let mut e = 0;
                    // process chunks of 4
                    while e + 4 <= n {
                        let pv =
                            f32x4::new([s.prev[e], s.prev[e + 1], s.prev[e + 2], s.prev[e + 3]]);
                        let av = f32x4::new([s.a[e], s.a[e + 1], s.a[e + 2], s.a[e + 3]]);
                        let vv = pv + av;
                        let arr = vv.to_array();
                        s.val[e] = arr[0];
                        s.val[e + 1] = arr[1];
                        s.val[e + 2] = arr[2];
                        s.val[e + 3] = arr[3];
                        e += 4;
                    }
                    // tail
                    while e < n {
                        s.val[e] = s.prev[e] + s.a[e];
                        e += 1;
                    }
                    let c_bf_forward = rdtsc();
                    // Block-based reverse suffix max to reduce long dependency chain.
                    // Choose block size 8 to match previous vector blocks.
                    let nval = window_len + 1;
                    if nval <= 32 {
                        // small: scalar reverse
                        s.best_from[window_len] = s.val[window_len];
                        s.cur[window_len] = 0.0_f32;
                        for e in (0..window_len).rev() {
                            let next_bf = s.best_from[e + 1];
                            let v = s.val[e];
                            s.best_from[e] = if v > next_bf { v } else { next_bf };
                            s.cur[e] = if next_bf.is_finite() {
                                next_bf - s.a[e]
                            } else {
                                f32::NEG_INFINITY
                            };
                        }
                    } else {
                        // larger: compute per-block maxima
                        let block = 8usize;
                        let blocks = (nval + block - 1) / block;
                        // compute maxima within each block into s.block_max[0..blocks]
                        // Use 4-wide SIMD to reduce the inner per-block max loop
                        let mut idx = 0usize;
                        let mut b = 0usize;
                        while idx < nval {
                            let end = (idx + block).min(nval);
                            let mut m = f32::NEG_INFINITY;
                            let len = end - idx;
                            if len >= 8 {
                                // full 8-element block: use two f32x4 lanes and elementwise max
                                let v0 = f32x4::new([
                                    s.val[idx],
                                    s.val[idx + 1],
                                    s.val[idx + 2],
                                    s.val[idx + 3],
                                ]);
                                let v1 = f32x4::new([
                                    s.val[idx + 4],
                                    s.val[idx + 5],
                                    s.val[idx + 6],
                                    s.val[idx + 7],
                                ]);
                                let mv = v0.max(v1);
                                let arr = mv.to_array();
                                let mut local_m = arr[0];
                                if arr[1] > local_m {
                                    local_m = arr[1];
                                }
                                if arr[2] > local_m {
                                    local_m = arr[2];
                                }
                                if arr[3] > local_m {
                                    local_m = arr[3];
                                }
                                m = local_m;
                            } else {
                                // tail block: scalar scan
                                for j in idx..end {
                                    let v = s.val[j];
                                    if v > m {
                                        m = v;
                                    }
                                }
                            }
                            s.block_max[b] = m;
                            idx += block;
                            b += 1;
                        }
                        // propagate suffix maxima across blocks
                        if blocks > 1 {
                            let mut bb = blocks - 1;
                            while bb > 0 {
                                let next = s.block_max[bb];
                                if next > s.block_max[bb - 1] {
                                    s.block_max[bb - 1] = next;
                                }
                                bb -= 1;
                            }
                        }
                        // fill best_from per-block using local reverse scan within block
                        // Do not compute `cur` here; compute it in a separate vectorized pass
                        idx = 0;
                        let mut bidx = 0;
                        while idx < nval {
                            let end = (idx + block).min(nval);
                            // local reverse scan, starting with suffix max from next block if any
                            let mut local_next = if bidx + 1 < blocks {
                                s.block_max[bidx + 1]
                            } else {
                                f32::NEG_INFINITY
                            };
                            for e in (idx..end).rev() {
                                let v = s.val[e];
                                let bf = if v > local_next { v } else { local_next };
                                s.best_from[e] = bf;
                                local_next = bf;
                            }
                            idx += block;
                            bidx += 1;
                        }
                        // ensure last element (nval-1) is set
                        if nval > 0 {
                            s.best_from[nval - 1] = s.val[nval - 1];
                        }

                        // Vectorized pass to compute cur[e] = best_from[e+1] - a[e]
                        if nval > 1 {
                            // compute up to nval-2 using vectorized 4-wide chunks
                            let n_minus1 = nval - 1;
                            let mut ee = 0usize;
                            while ee + 4 <= n_minus1 {
                                let bf_next = f32x4::new([
                                    s.best_from[ee + 1],
                                    s.best_from[ee + 2],
                                    s.best_from[ee + 3],
                                    s.best_from[ee + 4],
                                ]);
                                let av =
                                    f32x4::new([s.a[ee], s.a[ee + 1], s.a[ee + 2], s.a[ee + 3]]);
                                let bf_arr = bf_next.to_array();
                                let av_arr = av.to_array();
                                // per-lane finite check and subtraction
                                for lane in 0..4 {
                                    let bfv = bf_arr[lane];
                                    if bfv.is_finite() {
                                        s.cur[ee + lane] = bfv - av_arr[lane];
                                    } else {
                                        s.cur[ee + lane] = f32::NEG_INFINITY;
                                    }
                                }
                                ee += 4;
                            }
                            // tail
                            while ee < n_minus1 {
                                let bf_next = s.best_from[ee + 1];
                                s.cur[ee] = if bf_next.is_finite() {
                                    bf_next - s.a[ee]
                                } else {
                                    f32::NEG_INFINITY
                                };
                                ee += 1;
                            }
                            // last element
                            s.cur[nval - 1] = 0.0_f32;
                        } else if nval == 1 {
                            s.cur[0] = 0.0_f32;
                        }
                    }
                    let c_b1 = rdtsc();
                    phase_best_from_cycles += c_b1.wrapping_sub(c_b0);
                    phase_cur_cycles += c_b1.wrapping_sub(c_bf_forward);

                    if k == l - 1 {
                        // copy cur -> dp_lminus1 without creating simultaneous borrows of `s`
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                s.cur.as_ptr(),
                                s.dp_lminus1.as_mut_ptr(),
                                window_len + 1,
                            );
                        }
                        captured = true;
                    }

                    // rotate: move prev out, swap with cur, then put prev back to avoid
                    // multiple simultaneous mutable borrows of fields of `s`.
                    let mut tmp = std::mem::take(&mut s.prev);
                    std::mem::swap(&mut s.cur, &mut tmp);
                    s.prev = tmp;
                    let c_r1 = rdtsc();
                    // approximate rotate cycles as difference from previous c_b1
                    phase_rotate_cycles += c_r1.wrapping_sub(c_b1);
                }

                // accumulate elapsed time for this DP run (and per-phase)
                let elapsed_ns = dp_start.elapsed().as_nanos();
                let cycles_end = rdtsc();
                let dp_cycles = cycles_end.wrapping_sub(cycles_start);
                if window_len < s.instr_dp_time_ns.len() {
                    s.instr_dp_time_ns[window_len] += elapsed_ns;
                    s.instr_dp_cycles[window_len] += dp_cycles;
                    s.instr_dp_cycles_a[window_len] += phase_a_cycles;
                    s.instr_dp_cycles_best_from[window_len] += phase_best_from_cycles;
                    s.instr_dp_cycles_cur[window_len] += phase_cur_cycles;
                    s.instr_dp_cycles_rotate[window_len] += phase_rotate_cycles;
                    s.instr_dp_cycles_select[window_len] += phase_select_cycles;
                }

                // Select best first segment end using dp[l-1] (or prev if l==1)
                let dp_for_suffix_f32: &[f32] = if l >= 2 && captured {
                    &s.dp_lminus1[..=window_len]
                } else {
                    &s.prev[..=window_len]
                };

                // selection: find best end_rel using dp_for_suffix and a[]
                let t_sel = std::time::Instant::now();
                let c_sel0 = rdtsc();
                let mut best_val = f64::NEG_INFINITY;
                let mut best_end_rel = 0usize;
                // Chunked vectorized addition to reduce scalar ops; comparisons remain scalar
                let mut er = 1usize;
                while er + 4 <= window_len + 1 {
                    let vv = f32x4::new([
                        dp_for_suffix_f32[er] + s.a[er],
                        dp_for_suffix_f32[er + 1] + s.a[er + 1],
                        dp_for_suffix_f32[er + 2] + s.a[er + 2],
                        dp_for_suffix_f32[er + 3] + s.a[er + 3],
                    ]);
                    let arr = vv.to_array();
                    for lane in 0..4 {
                        let val = arr[lane] as f64;
                        let idx = er + lane;
                        if val > best_val + 1e-9 {
                            best_val = val;
                            best_end_rel = idx;
                        } else if (val - best_val).abs() <= 1e-9 && idx > best_end_rel {
                            best_end_rel = idx;
                        }
                    }
                    er += 4;
                }
                // tail
                while er <= window_len {
                    let val = (dp_for_suffix_f32[er] + s.a[er]) as f64;
                    if val > best_val + 1e-9 {
                        best_val = val;
                        best_end_rel = er;
                    } else if (val - best_val).abs() <= 1e-9 && er > best_end_rel {
                        best_end_rel = er;
                    }
                    er += 1;
                }
                let c_sel1 = rdtsc();
                phase_select_cycles += c_sel1.wrapping_sub(c_sel0);
                let sel_ns = t_sel.elapsed().as_nanos();
                if window_len < s.instr_dp_time_select_ns.len() {
                    s.instr_dp_time_select_ns[window_len] += sel_ns;
                }
                let best_total = if best_val.is_finite() {
                    best_val - (s.a[0] as f64)
                } else {
                    f64::NEG_INFINITY
                };

                if best_end_rel > 0 && best_total > 0.0 {
                    segments.push(Segment {
                        start: i,
                        end: i + best_end_rel,
                    });
                    i += best_end_rel;
                } else {
                    i += 1;
                }
            }); // end SCRATCH.with
        }
        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct MockOp {
        name: &'static str,
        gpu: bool,
        c: f64,
        t: usize,
    }
    impl Op for MockOp {
        fn name(&self) -> &str {
            self.name
        }
        fn gpu_capable(&self) -> bool {
            self.gpu
        }
        fn estimate(&self) -> (f64, usize) {
            (self.c, self.t)
        }
    }

    #[test]
    fn greedy_fuses_positive_benefit() {
        let ops: Vec<Arc<dyn Op>> = vec![
            Arc::new(MockOp {
                name: "a",
                gpu: true,
                c: 5.0,
                t: 0,
            }),
            Arc::new(MockOp {
                name: "b",
                gpu: true,
                c: 5.0,
                t: 0,
            }),
            Arc::new(MockOp {
                name: "c",
                gpu: false,
                c: 0.0,
                t: 0,
            }),
            Arc::new(MockOp {
                name: "d",
                gpu: true,
                c: 1.0,
                t: 1_000_000,
            }),
        ];
        let p = GreedyPartitioner {
            lookahead: 2,
            bytes_per_ms: 1_000_000.0,
            max_ops_in_fusion: 10,
        };
        let segs = p.partition(&ops);
        // Expect first two fused as one segment, last one is borderline (compute 1ms - transfer 1MB/1MB_per_ms = 0) -> not fused
        assert_eq!(segs, vec![Segment { start: 0, end: 2 }]);
    }

    #[test]
    fn greedy_caps_by_max_ops() {
        let mut ops: Vec<Arc<dyn Op>> = Vec::new();
        for _ in 0..200 {
            ops.push(Arc::new(MockOp {
                name: "x",
                gpu: true,
                c: 1.0,
                t: 0,
            }));
        }
        let p = GreedyPartitioner {
            lookahead: 2,
            bytes_per_ms: 1_000_000.0,
            max_ops_in_fusion: 50,
        };
        let segs = p.partition(&ops);
        // Should create multiple segments, none longer than 50
        assert!(segs.iter().all(|s| (s.end - s.start) <= 50));
    }
}
