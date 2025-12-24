# Test Coverage Analysis

## Executive Summary

The codebase has **550+ test functions** across 8 crates, with strong coverage in core areas (terminal emulation, actor scheduling, pixelflow-core) but notable gaps in newer or specialized modules.

**Overall Assessment:** Good foundation, but several critical areas need attention.

---

## Coverage by Crate

| Crate | Inline Tests | Integration Tests | Benchmarks | Assessment |
|-------|--------------|-------------------|------------|------------|
| **core-term** | 276 | 0 | 0 | ⭐⭐⭐⭐⭐ Excellent |
| **pixelflow-runtime** | 8 | 91 | 0 | ⭐⭐⭐⭐ Good |
| **pixelflow-core** | 5 | 118 | 1 | ⭐⭐⭐⭐ Good |
| **pixelflow-graphics** | 15 | 28 | 1 | ⭐⭐⭐ Moderate |
| **actor-scheduler** | 6 | 0 | 0 | ⭐⭐ Minimal |
| **actor-scheduler-macros** | 0 | 2 | 0 | ⭐⭐ Compilation only |
| **pixelflow-ml** | 3 | 0 | 0 | ⭐ Very limited |

---

## High Priority Gaps (Recommended Improvements)

### 1. **pixelflow-runtime/src/display/** — No Tests

**Location:** `pixelflow-runtime/src/display/`

**Impact:** High — This is the display driver layer handling Metal, X11, headless, and web backends.

**Files without tests:**
- `drivers/metal.rs` — macOS Metal graphics backend
- `drivers/x11.rs` — Linux X11 backend
- `drivers/headless.rs` — Headless rendering
- `drivers/web/mod.rs` and `web/ipc.rs` — WebAssembly backend
- `platform.rs` — Platform detection/abstraction
- `driver.rs` — Display driver trait
- `messages.rs` — Display message types
- `ops.rs` — Display operations

**Recommended Tests:**
```rust
// tests/display_driver_tests.rs

#[test]
fn headless_driver_renders_frame() {
    // Test that headless driver can accept and render a frame
}

#[test]
fn display_message_serialization() {
    // Test DisplayMessage variants serialize/deserialize correctly
}

#[test]
fn platform_detection_returns_valid_backend() {
    // Verify platform detection logic
}
```

---

### 2. **pixelflow-graphics/src/transform.rs** — No Tests

**Location:** `pixelflow-graphics/src/transform.rs:1-51`

**Impact:** Medium-High — Coordinate transformations (Scale, Translate) are foundational.

**Recommended Tests:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_doubles_coordinates() {
        let circle = Circle::new(0.0, 0.0, 1.0);
        let scaled = Scale { manifold: circle, factor: 2.0 };
        // At (2,0), scaled circle should evaluate same as original at (1,0)
    }

    #[test]
    fn translate_shifts_origin() {
        let circle = Circle::new(0.0, 0.0, 1.0);
        let translated = Translate { manifold: circle, offset: [1.0, 1.0] };
        // At (1,1), translated circle should evaluate same as original at (0,0)
    }

    #[test]
    fn scale_and_translate_compose() {
        // Test composition of transformations
    }
}
```

---

### 3. **pixelflow-graphics/src/image.rs** — No Tests

**Location:** `pixelflow-graphics/src/image.rs:1-34`

**Impact:** Medium — Image buffer with placeholder render implementation.

**Recommended Tests:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_new_creates_correct_buffer_size() {
        let img = Image::new(100, 50);
        assert_eq!(img.data.len(), 100 * 50 * 4); // RGBA
    }

    #[test]
    fn image_data_initializes_to_zero() {
        let img = Image::new(10, 10);
        assert!(img.data.iter().all(|&b| b == 0));
    }
}
```

---

### 4. **pixelflow-ml** — Theoretical Library Needs Practical Tests

**Location:** `pixelflow-ml/src/lib.rs`

**Impact:** Medium — Only 3 tests for a sophisticated ML/graphics unification library.

**Current Tests:**
- `test_elu_feature_positive` — Incomplete (doesn't verify output)
- `test_harmonic_attention_accumulate` — Basic smoke test
- `test_correspondence_doc` — Just checks compilation

**Recommended Tests:**
```rust
#[test]
fn elu_feature_positive_for_negative_input() {
    let f = EluFeature;
    // Verify ELU(x) + 1 > 0 for all x
}

#[test]
fn harmonic_attention_orthogonal_queries_independent() {
    // Queries in orthogonal directions should give independent results
}

#[test]
fn sh_feature_map_normalized() {
    // Verify SH projection produces normalized coefficients
}

#[test]
fn linear_attention_reset_clears_state() {
    let mut attn = LinearAttention::new(EluFeature, 1, 3);
    // Add some state, reset, verify cleared
}

#[test]
fn random_fourier_feature_dimension_correct() {
    let rff = RandomFourierFeature::new(vec![1.0, 2.0, 3.0]);
    assert_eq!(rff.num_features, 6); // 2x for sin/cos pairs
}
```

---

### 5. **actor-scheduler** — Needs Integration Tests

**Location:** `actor-scheduler/src/lib.rs`

**Impact:** Medium — Core scheduling infrastructure has only 6 unit tests.

**Current Coverage:**
- `verify_priority_ordering_contract` ✓
- `verify_data_lane_backpressure_contract` ✓
- `verify_actor_trait_contract` ✓
- Troupe pattern tests ✓

**Missing Tests:**
```rust
// tests/scheduler_stress_tests.rs

#[test]
fn high_contention_fairness() {
    // Multiple senders competing for control lane should all eventually succeed
}

#[test]
fn backoff_jitter_prevents_thundering_herd() {
    // Measure variance in send times under contention
}

#[test]
fn wake_handler_called_on_message_send() {
    // Verify custom wake handlers are invoked
}

#[test]
fn scheduler_handles_rapid_sender_drop() {
    // Create/drop many senders rapidly
}
```

---

### 6. **core-term/src/surface/** — Limited Tests

**Location:** `core-term/src/surface/`

**Impact:** Medium — Terminal surface abstraction with only manifold.rs having tests.

**Files needing tests:**
- `terminal.rs` — Terminal surface implementation
- `mod.rs` — Module organization

---

## Medium Priority Gaps

### 7. **Error Path Testing**

Many modules lack explicit error path tests. Consider:

```rust
#[test]
fn actor_handle_send_after_receiver_drop_returns_error() {
    let (tx, rx) = ActorScheduler::<(), (), ()>::new(10, 10);
    drop(rx);
    assert!(tx.send(Message::Data(())).is_err());
}

#[test]
fn backoff_returns_timeout_after_max_attempts() {
    // Verify backoff_with_jitter returns Err(Timeout) eventually
}
```

### 8. **Property-Based Testing Opportunities**

Consider adding `proptest` or `quickcheck` for:

- **pixelflow-core**: Field arithmetic properties (commutativity, associativity)
- **ANSI parsing**: Fuzzing with random byte sequences
- **Terminal emulator**: Random command sequences maintain invariants

### 9. **Benchmark Coverage**

Current benchmarks are in:
- `pixelflow-core/benches/core_benches.rs`
- `pixelflow-graphics/benches/graphics_benches.rs`

**Missing benchmarks:**
- Actor scheduler throughput under load
- Terminal emulator render performance
- Display driver frame latency

---

## Low Priority / Future Considerations

### 10. **CI Coverage Tracking**

Consider adding `cargo-tarpaulin` or `grcov` to CI for automated coverage reports.

### 11. **Documentation Tests**

Many doc comments have `ignore` markers. Consider converting to runnable doctests:

```rust
/// # Example
/// ```
/// use actor_scheduler::{ActorScheduler, Message};
/// let (tx, rx) = ActorScheduler::<i32, (), ()>::new(10, 100);
/// tx.send(Message::Data(42)).unwrap();
/// ```
```

### 12. **Snapshot/Golden Tests**

For rendering code, consider snapshot tests that compare output against known-good images.

---

## Summary of Recommendations

| Priority | Area | Effort | Impact |
|----------|------|--------|--------|
| 🔴 High | Display drivers | Medium | High |
| 🔴 High | transform.rs | Low | Medium |
| 🔴 High | image.rs | Low | Medium |
| 🟡 Medium | pixelflow-ml expansion | Medium | Medium |
| 🟡 Medium | actor-scheduler integration | Medium | Medium |
| 🟡 Medium | Error path tests | Low | Medium |
| 🟢 Low | Property-based tests | High | High |
| 🟢 Low | CI coverage tracking | Low | Medium |

---

## Immediate Action Items

1. Add inline tests to `pixelflow-graphics/src/transform.rs`
2. Add inline tests to `pixelflow-graphics/src/image.rs`
3. Create `pixelflow-runtime/tests/display_driver_tests.rs` for headless driver
4. Expand `pixelflow-ml` test coverage for `HarmonicAttention` and `LinearAttention`
5. Add `actor-scheduler/tests/stress_tests.rs` for concurrency edge cases
