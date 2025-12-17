# Bolt's Journal

## 2024-05-22 - Term Emulator Optimization
**Learning:** Terminal emulation involves frequent state checks in the hot loop (`print_char`). Identifying invariant state (like `ScreenContext` geometry during a single char print) allows hoisting context creation out of the loop, saving redundant struct constructions. Also, `slice::fill` and `copy_from_slice` are critical for bulk grid operations (`fill_region`, `resize`) instead of manual iteration.
**Action:** Always look for invariant state reconstruction in hot paths and hoist it. Use slice intrinsics for buffer manipulation.
