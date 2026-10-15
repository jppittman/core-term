# pixelflow-runtime/src/platform/macos

**Is.** The Cocoa end of the pull (DENOTATION §5): `MetalOps`, the `PlatformOps` that holds the
`NSApplication` and its windows, reports each window's extent in points and in device pixels (a
change of either is a new extent and a recompile) and presents the lent frame through a
`CAMetalLayer`. It runs on the main thread and carries no language.

## Owns
- The `NSApplication`, every `MacWindow` (`NSWindow`, `NSView`, `CAMetalLayer`) and the
  `PixelflowWindowDelegate` class, registered once.
- The event pump, `MetalOps::handle_os`: it hand-pumps `nextEventMatchingMask`, not
  `-[NSApplication run]`. Only the first wait blocks, and only on `Idle`; the queue then drains.
- Points versus pixels: `Surface` carries points (`width_px`, `height_px`: layout, input) and
  device pixels (`frame_width`, `frame_height`: the sample lattice) with the backing `scale`;
  `events::map_event` flips Y from AppKit's bottom-left origin.
- The queues from AppKit delegate callbacks to the pump (`CLOSED_WINDOWS`, `SCALE_CHANGES`), and
  spending the wake token the pump dequeues (`consume_wake_token`).
- The layer's pixel format, `RGBA8Unorm` (70), which is `PlatformPixel` on macOS, and the
  Objective-C FFI: `sys` is the raw runtime, `cocoa` the typed wrappers over it.

## Never
- No AppKit call off the main thread: nothing here spawns a thread, and the one cross-thread
  entry, `CocoaWaker::wake`, only enqueues onto the main dispatch queue.
- No send: nothing here holds an actor handle or calls `.send(`; events go into `DriverOut`.
- No keyboard or scroll event goes through `sendEvent:` (`event_type::is_spatial` gates it).
- No terminal logic, kernel or SIMD: nothing here imports `pixelflow_core` or `core-term`.

## Contract
`mod.rs` declares `cocoa`, `events`, `platform`, `sys`, `window` and re-exports `MetalOps`.
`platform/mod.rs` selects `ActivePlatform = PlatformActor<macos::MetalOps>`;
`display/drivers/metal.rs` names `MetalDisplayDriver`. `MetalOps` implements `PlatformOps`:
`handle_data` (`Present`), `handle_control` (window, cursor, bell, clipboard, fullscreen),
`handle_management` (`Create`, `Destroy`), `handle_os` (the pump). `Copy{Primary}` is dropped.

## Enforced by
- `platform/mod.rs` gates the module whole (`#[cfg(target_os = "macos")] pub mod macos;`); the
  baseline of `scripts/check-cfg-encapsulation.sh` has no entry under this path.
- `pixelflow-runtime/src/display/ops.rs`: `PlatformOps` methods take a `DriverOut` and return no
  event, so the no-send line is the trait's signature; desloppify `effects-are-returned`.
- `xtask launch-stability-check` (job "macOS launch stability (informational, non-blocking)"):
  15 `open` launches, each watched 25s, then a scan for crash reports; the only check that runs
  the pump under LaunchServices, and it does not block merges until it has a green history.
- `window.rs` `abi_alignment` (`CGSize`, `MTLSize` layout), in the macOS `test` job.

## Behind
- `objc.rs`, a second binding layer with its own `Id`, is declared nowhere and compiled by
  nothing; `sys` and `cocoa` are the live pair.
- The `cocoa.rs` header promises no `unsafe` in driver logic; `window.rs` (17 `unsafe` blocks)
  and `platform.rs` (4) call `sys::send` directly. No code asserts the main thread
  (`unsafe impl Send for MetalOps` rests on the `[main]` actor placement alone).
- A failed `DisplayMgmt::Create` is `eprintln!`ed and no event reaches the engine; the
  `NSAutoreleasePool` in `MetalOps::new` is never drained.
- `map_key` and `map_modifiers` have no test (`map_event` takes an `NSEvent`); the window tests and
  `tests/macos_integration_tests.rs` need a window server and are `#[ignore]`d.

## Decided in
`docs/designs/pixelflow-runtime-engine-mesh-migration.md` §5.1 (`PlatformOps` returns its events);
the `CocoaWaker` header in `pixelflow-runtime/src/platform/waker.rs` (the `dispatch_async` hop);
CLAUDE.md "Platform Notes".
