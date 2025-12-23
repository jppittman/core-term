# Performance Analysis Report

This document identifies performance anti-patterns, potential N+1 issues, and inefficient algorithms found in the core-term codebase.

## Executive Summary

The codebase is generally well-designed with deliberate performance considerations (SIMD vectorization, lazy evaluation, zero-copy frames). However, several opportunities for optimization exist, particularly in the terminal grid management and rendering pipeline.

---

## 🔴 HIGH Priority Issues

### 1. Arc Copy-on-Write Overhead Per Character Write

**Status:** ✅ **NOT AN ISSUE** - Arc::make_mut is optimal

**Location:** `core-term/src/term/screen.rs:236, 396, 437, 644`

**Analysis:** `Arc::make_mut()` only clones if `refcount > 1`. In practice:
- If renderer has released the snapshot: refcount = 1, **no clone**
- If renderer holds snapshot: clones **once per row**, not per character
- Subsequent writes to same row are free until next snapshot

This is the correct COW behavior - you only pay for divergence when you actually need it.

---

### 2. Full Grid Traversal Every Frame

**Status:** ✅ **FIXED**

**Location:** `core-term/src/surface/grid.rs`

**Solution:** Added `GridBuffer::update_from_snapshot()` that skips clean lines:
```rust
// Only process dirty lines
if !line.is_dirty {
    continue;
}
```

**Impact:** For 80×24 terminal with 1 dirty line:
- Before: 1,920 cell conversions per frame
- After: 80 cell conversions (24x reduction)

---

### 3. SIMD Sequential Field Creation Suboptimal

**Status:** ✅ **FIXED**

**Location:** `pixelflow-graphics/src/render/rasterizer/mod.rs:89`

**Solution:** Use `Field::sequential(fx)` directly:
```rust
// Before: Field::from(fx) + Field::from(0.0)
// After:
let xs = Field::sequential(fx);
```

Made `Field::sequential` public in pixelflow-core for this use case.

---

## 🟡 MEDIUM Priority Issues

### 4. Thread Creation Per Frame

**Location:** `pixelflow-runtime/src/render_pool.rs:58-64`

**Problem:** Uses `std::thread::scope()` creating new threads for every parallel render:

```rust
// render_pool.rs:58
std::thread::scope(|s| {
    for (chunk, start_y, end_y) in buffer_chunks {
        s.spawn(move || { ... });
    }
});
```

**Impact:**
- Thread creation/destruction overhead (~10-20μs per thread)
- CPU cache thrashing as threads don't maintain affinity
- Stack allocation per spawn

**Recommendation:**
- Use a persistent thread pool (e.g., `rayon` or custom pool)
- Pin threads to cores for consistent cache behavior
- Reuse worker threads across frames

---

### 5. Range Object Creation in UTF-8 Decoder Hot Path

**Location:** `core-term/src/ansi/lexer.rs:86-94`

**Problem:** Creates multiple `RangeInclusive` objects per byte in the decode hot path:

```rust
// lexer.rs:86-94 - called for EVERY byte
fn decode_first_byte(&mut self, byte: u8) -> Utf8DecodeResult {
    let utf8_ascii_range: std::ops::RangeInclusive<u8> = 0x00..=UTF8_ASCII_MAX;
    let utf8_invalid_early_start_range: std::ops::RangeInclusive<u8> = ...;
    let utf8_2_byte_start_range: std::ops::RangeInclusive<u8> = ...;
    let utf8_3_byte_start_range: std::ops::RangeInclusive<u8> = ...;
    // ...
}
```

**Impact:** While Rust likely optimizes this, it's conceptually wasteful. For high-throughput terminal I/O (MB/s), every nanosecond matters.

**Recommendation:**
- Declare ranges as `const` at module level (already partially done with individual bounds)
- Use direct comparison: `if byte <= UTF8_ASCII_MAX` instead of `utf8_ascii_range.contains(&byte)`
- Consider a lookup table for byte classification

---

### 6. Actor Scheduler Backoff Overhead

**Location:** `actor-scheduler/src/lib.rs:280-296`

**Problem:** Uses `Instant::now().elapsed()` for jitter hash in backoff:

```rust
// lib.rs:290-291
let now = Instant::now();
let hash = (now.elapsed().as_nanos() as u64).wrapping_mul(JITTER_HASH_CONSTANT);
```

**Impact:**
- Syscall overhead to get current time
- `elapsed()` called immediately after `now()` is meaningless (always ~0)
- Called in tight retry loop

**Recommendation:**
- Use a thread-local PRNG or simple counter for jitter
- Or use `Instant::now().as_nanos()` directly if some time-based variation is desired

---

### 7. Potential Vec Reallocation in Token Collection

**Location:** `core-term/src/ansi/lexer.rs:183-184, 288-290`

**Problem:** `AnsiLexer` uses a `Vec<AnsiToken>` that grows dynamically:

```rust
pub struct AnsiLexer {
    tokens: Vec<AnsiToken>,  // No capacity hint
    // ...
}
```

**Impact:** For large input bursts, the Vec may reallocate multiple times as it grows.

**Recommendation:**
- Pre-allocate with `Vec::with_capacity()` based on expected input size
- Consider using a ring buffer or bounded queue
- Reuse the Vec across calls via `clear()` instead of `take()`

---

## 🟢 LOW Priority / Informational

### 8. Scrollback Buffer Memory Growth

**Location:** `core-term/src/term/screen.rs:67-69`

**Issue:** Unbounded VecDeque with configurable limit. Memory grows linearly with scroll history.

**Mitigation:** Already has `scrollback_limit` from config. No action needed unless memory becomes an issue.

---

### 9. Bounds Clamping Instead of Assertions

**Location:** `core-term/src/surface/grid.rs:126-128`

**Issue:** Out-of-bounds access is silently clamped:

```rust
pub fn get(&self, col: usize, row: usize) -> &Cell {
    let idx = row * self.cols + col;
    &self.cells[idx.min(self.cells.len() - 1)]  // Silent clamp!
}
```

**Impact:** This hides bugs - wrong cell returned instead of panic.

**Recommendation:** Use `debug_assert!` to catch bounds violations in debug builds while keeping release performance.

---

### 10. Manifold Evaluation Depth

**Location:** `pixelflow-core/src/` (manifold trait implementations)

**Issue:** Each pixel triggers recursive manifold evaluation. Deep combinator nesting = repeated coordinate transforms.

**Mitigation:** Monomorphization and `#[inline(always)]` help significantly. The design is intentional for flexibility.

---

## Algorithmic Complexity Summary

| Operation | Current | Optimal | Gap |
|-----------|---------|---------|-----|
| Single char write | O(row_width) | O(1) | Arc::make_mut clones row |
| Frame render (grid) | O(cols × rows) | O(dirty_cells) | No dirty tracking used |
| UTF-8 decode | O(1) per byte | O(1) | Minor: range object creation |
| Parallel render setup | O(threads) | O(1) | Thread creation per frame |

---

## Recommended Priority Order

1. **Use dirty flags in GridBuffer::from_snapshot()** - Quick win, significant impact
2. **Batch Arc mutations for terminal writes** - Medium effort, high impact for I/O-heavy workloads
3. **Thread pool for render_pool** - Medium effort, consistent frame times
4. **Optimize UTF-8 decoder** - Low effort, matters for high-throughput scenarios
5. **Fix Instant backoff** - Trivial fix

---

## Profiling Recommendations

To validate these findings, profile with:

```bash
# Build with profiling symbols
RUSTFLAGS="-C debuginfo=2" cargo build --release

# Use flamegraph
cargo flamegraph --bin core-term -- [your test case]

# Key metrics to measure:
# - Time in GridBuffer::from_snapshot()
# - Count of Arc::make_mut() clones
# - render_pool thread spawn overhead
# - Per-frame latency distribution
```

Focus profiling on:
1. `cat /dev/urandom | head -c 1000000` - stress test terminal I/O
2. `seq 1 100000` - rapid line output
3. Interactive typing - latency-sensitive path
