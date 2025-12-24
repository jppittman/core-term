## 2025-05-23 - Performance Analysis Review
**Learning:** `PERFORMANCE_ANALYSIS.md` is a critical resource. It pointed directly to thread creation as a bottleneck.
**Action:** Always check for `PERFORMANCE_ANALYSIS.md` or similar documents before starting.

## 2025-05-23 - Dependency Management
**Learning:** `rayon` is present in `Cargo.lock` but not in `pixelflow-runtime`'s `Cargo.toml`.
**Action:** Dependencies must be explicitly added to the crate's `Cargo.toml` to be used.
## 2025-05-23 - Dependency Management (Rayon Rejection)
**Learning:** Even if a dependency is in the lockfile, adding it to a crate might be rejected if the user wants to avoid heavy dependencies for simple tasks. 'Chesterton's fence' applies to missing dependencies too.
**Action:** Always ask before adding a dependency, even if it seems 'available'. Preference is for lightweight, native solutions.
