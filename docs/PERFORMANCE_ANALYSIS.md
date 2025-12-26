# Performance Analysis Report

This document identifies performance anti-patterns, N+1 query patterns, and inefficient algorithms in the core-term codebase.

## Executive Summary

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Font/Glyph Handling | 1 | 2 | 4 | 3 |
| Actor Messaging | 2 | 1 | 3 | 3 |
| ANSI Parsing | 1 | 1 | 2 | 2 |
| Rendering Pipeline | 0 | 2 | 3 | 2 |
| **Total** | **4** | **6** | **12** | **10** |

---

## Critical Issues

### 1. N+1 CMAP Lookups in Text Rendering

**Files:**
- `pixelflow-graphics/src/fonts/cache.rs:284-317`
- `pixelflow-graphics/src/fonts/text.rs:22-57`
- `pixelflow-graphics/src/fonts/ttf.rs:743-778`

**Issue:** For each character in text rendering, the code performs **4 separate CMAP lookups**:

```rust
// CachedText::new loop (cache.rs:292-312)
for ch in text.chars() {
    cursor_x += font.kern_scaled(prev, ch, size);  // 2 lookups
    if let Some(cached) = cache.get(font, ch, size) {  // 1 lookup
    if let Some(adv) = font.advance_scaled(ch, size) {  // 1 lookup
}
```

Each CMAP lookup is O(n) linear search through character ranges.

**Impact:** Rendering a 1000-character terminal line performs 4000+ linear range searches.

**Fix:** Cache glyph ID after first lookup, use batch APIs, implement binary search for CMAP.

---

### 2. Heap Allocation on Every PTY Read

**File:** `core-term/src/io/event_monitor_actor/read_thread/mod.rs:83`

```rust
let data = buf[..n].to_vec();  // New heap allocation per read!
```

**Impact:** Every PTY read (up to 4KB) creates a new Vec allocation. For high-bandwidth I/O, this causes allocator contention and cache misses.

**Fix:** Use a ring buffer or pre-allocated pool of buffers.

---

### 3. Blocking Thread Joins in Drop (Shutdown Hang)

**Files:**
- `core-term/src/io/event_monitor_actor/read_thread/mod.rs:115`
- `core-term/src/io/event_monitor_actor/parser_thread/mod.rs:55`
- `core-term/src/io/event_monitor_actor/write_thread/mod.rs:50`

```rust
impl Drop for ReadThread {
    fn drop(&mut self) {
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();  // BLOCKING with no timeout!
        }
    }
}
```

**Impact:** If a thread is stuck in a blocking syscall, the entire application hangs on shutdown.

**Fix:** Add timeout to joins, or use non-blocking shutdown signals.

---

### 4. Cache Key Missing Font Identity (Correctness Bug)

**File:** `pixelflow-graphics/src/fonts/cache.rs:140-142`

```rust
struct CacheKey {
    codepoint: u32,     // Missing: font identifier!
    size_bucket: usize,
}
```

**Impact:** If the same cache is used with multiple fonts, glyphs from different fonts collide silently.

**Fix:** Add font identifier (pointer hash or font ID) to CacheKey.

---

## High Priority Issues

### 5. Linear CMAP Search Instead of Binary

**File:** `pixelflow-graphics/src/fonts/ttf.rs:569-605`

```rust
// Fmt4: Linear search O(n)
(0..n).find_map(|i| {
    let end = R(*d, 14 + i * 2).u16()?;
    // ...
})
```

**Impact:** CMAP ranges are sorted - binary search would be O(log n) instead of O(n).

---

### 6. Clone in Cache Hit Path

**File:** `pixelflow-graphics/src/fonts/cache.rs:192`

```rust
if let Some(cached) = self.entries.get(&key) {
    return Some(cached.clone());  // Clone on every hit!
}
```

**Impact:** Every cache hit clones the CachedGlyph struct (includes Arc increment).

**Fix:** Return `&CachedGlyph` or `Arc<CachedGlyph>` directly.

---

### 7. Conservative Burst Limits

**File:** `core-term/src/io/event_monitor_actor/mod.rs:66`

```rust
ActorScheduler::<...>::new(
    10,  // burst limit: too low
    64,  // buffer size: too small
);
```

**Impact:** Causes excessive context switching between read and parser threads.

---

### 8. Inefficient UTF-8 Encoding in OSC/DCS Parsing

**File:** `core-term/src/ansi/parser.rs:435-439`

```rust
AnsiToken::Print(c) => {
    let mut buf = [0; 4];
    c.encode_utf8(&mut buf)
        .as_bytes()
        .iter()
        .for_each(|&b| self.add_string_byte(b));
}
```

**Impact:** Every printable character in escape sequences triggers stack allocation, UTF-8 encoding, iterator creation, and per-byte function calls.

---

## Medium Priority Issues

### 9. Compound Glyph Children Recompiled on Each Access

**File:** `pixelflow-graphics/src/fonts/ttf.rs:917-918`

No memoization of compound glyph parsing - children are recompiled every time.

---

### 10. No Texture Atlasing

**File:** `pixelflow-graphics/src/fonts/cache.rs:78-99`

Each cached glyph gets its own independent texture allocation, causing memory fragmentation.

---

### 11. 4x Vec Allocation in Color Baking

**File:** `pixelflow-graphics/src/baked.rs:64-67`

```rust
let mut r_data = Vec::with_capacity(width * height);
let mut g_data = Vec::with_capacity(width * height);
let mut b_data = Vec::with_capacity(width * height);
let mut a_data = Vec::with_capacity(width * height);
```

Could use SIMD extraction or interleaved buffer.

---

### 12. Nested flat_map with Vec Allocation

**File:** `pixelflow-graphics/src/fonts/ttf.rs:932-943`

```rust
.flat_map(|(i, &(x, y, on))| {
    if !on && !non {
        vec![(x, y, on), ((x + nx) / 2.0, (y + ny) / 2.0, true)]
    } else {
        vec![(x, y, on)]
    }
})
```

Creates Vec per point during glyph parsing.

---

### 13. Busy-Loop with keep_working Flag

**File:** `actor-scheduler/src/lib.rs:575-639`

Inner scheduler loop can spin without yielding when messages arrive faster than processing.

---

### 14. Two Sequential Loops in ANSI Processing

**File:** `core-term/src/ansi/mod.rs:59-70`

```rust
for byte in bytes {
    self.lexer.process_byte(*byte);  // Loop 1
}
for token in tokens {                 // Loop 2
    self.parser.process_token(token);
}
```

Could be single-pass with pull-based token consumption.

---

### 15. Missing #[inline] on Font API Methods

**File:** `pixelflow-graphics/src/fonts/ttf.rs:743-776`

Public methods `glyph()`, `advance()`, `kern()` are called per-character and lack `#[inline]` attributes.

---

### 16. Unused Message Type Overhead

**File:** `core-term/src/io/event_monitor_actor/parser_thread/mod.rs:10-15`

NoControl/NoManagement types allocate channel slots that are never used.

---

### 17. State Enum Cloning During String Entry

**File:** `core-term/src/ansi/parser.rs:251,256`

State enum cloned on every OSC/DCS/PM/APC entry; could use discriminant index.

---

### 18. Redundant Calculations in Hot Loop

**File:** `pixelflow-graphics/src/fonts/cache.rs:289-290`

`size_bucket()` and scale division computed per-character instead of before loop.

---

### 19. Double Clone on Cache Insert

**File:** `pixelflow-graphics/src/fonts/cache.rs:198`

```rust
self.entries.insert(key, cached.clone());
Some(cached)
```

Clones before and after insert.

---

### 20. Expensive Jitter Calculation

**File:** `actor-scheduler/src/lib.rs:326-327`

`Instant::now()` called on every retry, then immediately `elapsed()` is queried (near-zero).

---

## Low Priority Issues

### 21. Manifold Clone in Parallel Dispatch

**File:** `pixelflow-graphics/src/render/rasterizer/parallel.rs:77`

Necessary for thread safety but worth noting for complex manifold types.

---

### 22. Kerning Never Cached

**File:** `pixelflow-graphics/src/fonts/cache.rs:295`

Same character pairs ("Th", "ng") computed repeatedly.

---

### 23. No Batch Glyph API

**File:** `pixelflow-graphics/src/fonts/ttf.rs:743-778`

Missing `glyphs_batch()`, `advances_batch()` for SIMD/prefetch optimization.

---

### 24. Pixel-by-Pixel Coverage Extraction

**File:** `pixelflow-graphics/src/fonts/cache.rs:84-90`

Per-pixel bitwise ops and division; could use SIMD batch conversion.

---

### 25. Small Fixed Buffer Sizes

**File:** `actor-scheduler/src/lib.rs:289`

`CONTROL_MGMT_BUFFER_SIZE = 128` is inflexible for different actor patterns.

---

### 26. Redundant Control Polling

**File:** `actor-scheduler/src/lib.rs:585,606`

Control messages polled twice per park cycle.

---

### 27. Weak Doorbell with try_send

**File:** `actor-scheduler/src/lib.rs:458`

Drops wake signal if channel full (by design, but worth noting).

---

### 28. Vec Allocation in Frame Convert

**File:** `pixelflow-graphics/src/render/frame.rs:49`

Per-frame allocation, acceptable but could reuse buffers.

---

### 29. Unconditional finalize() Call

**File:** `core-term/src/ansi/mod.rs:62-63`

Called even when no UTF-8 sequence in progress.

---

### 30. Pre-allocation Size Mismatch

**File:** `core-term/src/ansi/parser.rs:59`

String buffer pre-allocated at `MAX_OSC_LEN / 4` assumes 4-byte UTF-8, but most OSC is ASCII.

---

## Positive Findings

- **358 instances of `#[inline(always)]`** in pixelflow-core
- **All `eval_raw` hot path functions properly inlined**
- **No Mutex/RwLock in rendering pipeline**
- **Smart short-circuit evaluation** in Select combinators
- **Well-designed SIMD + scalar fallback** in rasterizer
- **Efficient byte-by-byte state machine** in ANSI lexer
- **Good use of `mem::take()` for ownership transfers**
- **Fixed 4-byte UTF-8 decoder buffer** (no dynamic allocation)

---

## Recommended Priority Order

1. **Critical:** Fix CMAP N+1 lookups - biggest performance win
2. **Critical:** Fix cache key to include font identity (correctness)
3. **Critical:** Add timeout to blocking thread joins (stability)
4. **High:** Implement binary search for CMAP tables
5. **High:** Eliminate heap allocation per PTY read
6. **High:** Return Arc reference from cache instead of cloning
7. **Medium:** Add texture atlasing for glyph cache
8. **Medium:** Add `#[inline]` to font API methods
9. **Medium:** Increase burst limits in parser actor
10. **Low:** Batch APIs and SIMD optimizations
