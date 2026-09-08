//! Proves `Dag::scratch()` + `Node::descendants_in` amortize to zero
//! allocations across repeated walks — the whole reason `Scratch` exists
//! over the simpler `Node::descendants()`.
//!
//! Needs a process-wide `#[global_allocator]` to count, which is why this
//! lives in its own integration-test binary rather than `dag.rs`'s unit
//! test module: `cargo test` runs a `--lib` binary's tests concurrently by
//! default, and unrelated tests allocating on other threads would pollute a
//! shared counter. A dedicated binary is its own process — nothing else
//! here is allocating into the count.

use pixelflow_ir::{Builder, Rooted};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: Counting = Counting;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
enum Op {
    Var(&'static str),
    Add,
    Mul,
}

fn build() -> Rooted<Op> {
    let mut b = Builder::new();
    let x = b.intern(Op::Var("x"), &[]);
    let y = b.intern(Op::Var("y"), &[]);
    let add = b.intern(Op::Add, &[x, y]);
    let root = b.intern(Op::Mul, &[add, x]);
    b.finish(&[root])
}

#[test]
fn repeated_walks_do_not_allocate() {
    let g = build();
    let mut sc = g.scratch();
    let mut sink = 0usize;

    // Measure 1000-call windows until two *consecutive* ones cost exactly
    // the same, instead of asserting a literal zero (which bakes in one
    // platform's allocator behavior) or a fixed warmup count (which just
    // moves the guess elsewhere). Observed on the macOS runner: a 64-call
    // warmup wasn't enough -- window 1 still cost 5 allocations, window 2
    // cost 0 -- i.e. some one-time, per-process allocator setup lands
    // *inside* the first measured window rather than before it, and there
    // is no fixed call count that's guaranteed to precede it. Retrying
    // until consecutive windows agree handles that regardless of how many
    // calls it takes to trigger, on any platform.
    let mut previous_window: Option<usize> = None;
    let mut steady_state = None;
    for _ in 0..50 {
        let before = ALLOCS.load(Ordering::Relaxed);
        for _ in 0..1000 {
            sink += g.entry().descendants_in(&mut sc).count();
        }
        let window = ALLOCS.load(Ordering::Relaxed) - before;
        if previous_window == Some(window) {
            steady_state = Some(window);
            break;
        }
        previous_window = Some(window);
    }
    let steady_state = steady_state.unwrap_or_else(|| {
        panic!("descendants_in's allocations never stabilized across 50 windows of 1000 calls each")
    });

    // The allocating variant, for contrast: it should cost meaningfully
    // more over the same number of calls than the reused-scratch steady
    // state -- whatever that steady state number turns out to be on this
    // platform.
    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..1000 {
        sink += g.entry().descendants().count();
    }
    let allocating = ALLOCS.load(Ordering::Relaxed) - before;
    assert!(
        allocating > steady_state + 100,
        "descendants() (fresh alloc per call) should cost far more than descendants_in \
         (reused scratch): allocating={allocating}, steady_state={steady_state}"
    );
    assert!(sink > 0);
}
