# pixelflow-runtime/src/display/drivers

**Is.** The seam where the build's display driver is named: one cfg, set once by
`build.rs`, picks the driver that holds the OS window the lattice's extent is
taken from (DENOTATION §5). The module holds no driver logic of its own; a driver
is a `PlatformOps` behind `PlatformActor` behind `DriverActor`, and this directory
only exports the one the cfg selects.

## Owns
- Which driver a build exports: `use_cocoa_display` exports `MetalDisplayDriver`,
  `use_web_display` exports `WebDisplayDriver`, and nothing is exported for
  `use_x11_display` (below). Each is a single-line `#[cfg]` dispatch item or a
  whole-module gate in `mod.rs`.
- The macOS driver's type, an alias with no code (`metal.rs`).

## Never
- No Cocoa, X11 or JS call here: `metal.rs` and `mod.rs` name no `objc`, `x11` or
  `cocoa` item; platform calls live in `platform/` and `drivers/web/`.
- No driver logic in an alias: `MetalDisplayDriver` adds no method, field or `impl`.
- No driver cfg without a recorded CI job (`scripts/check-driver-cfg-coverage.sh`).
- No new driver as a copy of an old one: a driver is another `impl PlatformOps`,
  wrapped by the existing `PlatformActor` and `DriverActor`.

## Contract
`mod.rs` declares `metal` and re-exports `MetalDisplayDriver` under `use_cocoa_display`;
declares `web` (own `intent.md`) and re-exports `WebDisplayDriver` under `use_web_display`.
`display/mod.rs` re-exports both again. Selection is the cfg `build.rs` emits
(`DISPLAY_DRIVER`, else the `display_*` features, else `target_os`); the `x11`
lines in `mod.rs` are commented out.

## Enforced by
- `scripts/check-cfg-encapsulation.sh` (job `cfg-encapsulation`): a cfg may gate a
  whole module or be a one-line dispatch item in `mod.rs`; this directory has no
  baselined violation in `scripts/cfg_encapsulation_baseline.txt`.
- `scripts/check-driver-cfg-coverage.sh` (job "Display driver CI coverage"): a
  `use_*_display` cfg declared in a `build.rs` with no ledger entry fails the build.
- `scripts/check-wasm-status.sh` against `scripts/wasm_status_baseline.txt`, and
  `scripts/test_wasm_status.py` (job "WASM status reporting regression"): the web
  driver's build failure is held at its baseline, in both directions.
- The type system: a driver that does not implement `PlatformOps` cannot be wrapped
  by `PlatformActor`. `use_cocoa_display` is covered by building on `macos-latest`,
  `use_x11_display` on `ubuntu-latest`; no test names `MetalDisplayDriver`.

## Behind
- The Linux driver is not here. `platform/linux` (`LinuxOps`) is selected in
  `platform/mod.rs` by `target_os = "linux"`, not by `use_x11_display`, and the
  `X11DisplayDriver` lines in `mod.rs` and `display/mod.rs` are dead comments.
  The Wayland plan (2026-09-03; spike done, phases 2 to 5 unbuilt) moves it to the cfg.
- `MetalDisplayDriver` duplicates `platform::PlatformDriver` on macOS, and nothing
  names it but the re-exports.
- The web driver does not build. It implements the legacy `DisplayDriver` trait
  (`display/driver.rs`), imports a `pixelflow_render` crate that is not in the
  workspace, and has no wasm32 JIT backend under it; the wasm status baseline lists
  it as "pixelflow-runtime web driver".

## Decided in
`docs/runtime/2026-09-03-wayland-driver.md` (the seam, the coverage ledger),
`docs/runtime/pixelflow-runtime-engine-mesh-migration.md` §5.1 (ops decide, the
adapter sends); CLAUDE.md "Platform cfg is encapsulation, not sprinkle" and "Platform Notes".
