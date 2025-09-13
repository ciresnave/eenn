use eenn::partitioner::Op;
use eenn::{GreedyPartitioner, Partitioner, Segment};
use std::sync::Arc;

struct MockOp {
    gpu: bool,
    c: f64,
    t: usize,
}
impl Op for MockOp {
    fn name(&self) -> &str {
        "m"
    }
    fn gpu_capable(&self) -> bool {
        self.gpu
    }
    fn estimate(&self) -> (f64, usize) {
        (self.c, self.t)
    }
}

#[test]
fn lookahead_allows_helpful_prefix() {
    // Sequence: op0 (small negative), op1 (large positive)
    // op0: compute 0.1ms, transfer 200_000 bytes
    // op1: compute 5.0ms, transfer 0
    let ops: Vec<Arc<dyn Op>> = vec![
        Arc::new(MockOp {
            gpu: true,
            c: 0.1,
            t: 200_000,
        }) as Arc<dyn Op>,
        Arc::new(MockOp {
            gpu: true,
            c: 5.0,
            t: 0,
        }) as Arc<dyn Op>,
    ];

    // bytes_per_ms such that op0 alone is negative: 0.1 - 200k/1_000_000 = -0.099
    let p_no_lookahead = GreedyPartitioner {
        lookahead: 1,
        bytes_per_ms: 1_000_000.0,
        max_ops_in_fusion: 10,
    };
    let segs0 = p_no_lookahead.partition(&ops);
    // with no lookahead it should not fuse op0 with op1; op1 itself is positive and will be fused alone
    assert_eq!(segs0, vec![Segment { start: 1, end: 2 }]);

    // with lookahead = 2, it should accept fusing op0+op1 because combined benefit is positive
    let p_lookahead = GreedyPartitioner {
        lookahead: 2,
        bytes_per_ms: 1_000_000.0,
        max_ops_in_fusion: 10,
    };
    let segs1 = p_lookahead.partition(&ops);
    assert_eq!(segs1, vec![Segment { start: 0, end: 2 }]);
}
