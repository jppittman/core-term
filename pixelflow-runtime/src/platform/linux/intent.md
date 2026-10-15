# pixelflow-runtime/src/platform/linux

**Is.** The X11 end of the pull: one OS window whose size is the lattice's extent
(a resize is a new extent and a recompile, DENOTATION §5), a frame lent to it and
uploaded whole by `XPutImage`, and the server's events as `DisplayEvent`s. It is
`PlatformOps` for Xlib and nothing else; the pixel is `Bgra8`.

## Owns
- The X connection: `X11Window`, `XInitThreads` before `XOpenDisplay`, and teardown
  order in `Drop` (clear the waker target, then destroy the window).
- X to the runtime's vocabulary: keysym to `KeySymbol` (the keysym is the key, the
  typed text only what it typed), state bits to `Modifiers`, buttons 4-7 to
  `MouseScroll`, `ConfigureNotify` to `Resized`.
- The selection protocol: owning CLIPBOARD and PRIMARY, answering `SelectionRequest`,
  and one `PasteData` per `request_paste` (empty when the owner refuses).
- The scale factor (`Xft.dpi / 96`, read once) and the poll: `handle_os` blocks in
  `XNextEvent` when `Idle`; `X11Waker` interrupts it from another thread.

## Never
- No send to the engine: no handle, channel or `Message` here. Events leave through
  `DriverOut` (`out.event`, `out.blitted`).
- No xlib call outside this directory except `X11Waker` (`platform/waker.rs`).
- No frame allocation: `present` wraps the frame's own memory in an `XImage`,
  detaches it before `XDestroyImage`, and hands the frame back.
- No terminal logic: keys are reported as keys, never as escape sequences.
- No platform `#[cfg]` on an item: the directory is gated whole at `mod linux;`.
- No Wayland branch: a second driver is a second `impl PlatformOps`.

## Contract
`mod.rs` re-exports `LinuxOps` and `set_shared_waker`. `platform.rs` is the
`PlatformOps` impl (trait in `display/ops.rs`); `window.rs` is `X11Window` and
`SelectionAtoms`; `events.rs` is `map_event`. Present returns the frame through
`blitted`; `RequestRedraw` is a no-op, every present being a whole frame; Create emits
`WindowCreated`. `ActivePlatform` is `PlatformActor<LinuxOps>` (`platform/mod.rs`).

## Enforced by
- `PlatformOps` (`display/ops.rs`): events are returned through `&mut DriverOut`, so
  the no-send line is the signature; `LinuxOps::new` takes no engine handle.
- `scripts/check-cfg-encapsulation.sh`: `scripts/cfg_encapsulation_baseline.txt` has no
  entry under `platform/linux/`, and a new one fails.
- `scripts/check-driver-cfg-coverage.sh`: `use_x11_display` is built by ubuntu-latest.
- `events.rs` `mod tests`: the keysym mapping (named keys, chords, keypad, Unicode).
- Nothing drives `LinuxOps` without an X server: `tests/platform_actor_tests.rs`
  covers `PlatformActor` over a mock `PlatformOps`, not this module.

## Behind
- The module is selected by `target_os = "linux"`, `X11Waker` by `use_x11_display`: a
  Linux build with `DISPLAY_DRIVER` set to another driver does not compile. `build.rs`
  picks x11 on Linux unless overridden (Wayland plan, "the seam").
- No Wayland driver exists: no `wayland` in any `.rs`, `.toml`, workflow or script; of
  the plan's five phases only the spike is done (`2026-09-03-wayland-driver.md`).
- A failed `Create` is logged and dropped, so the app is never told no window exists;
  a failed `XCreateImage` still reports `blitted`; `set_shared_waker` discards a second
  set with `.ok()`. No plan closes these.
- `platform.rs`'s header names the deleted `X11DisplayDriver`; `LinuxPixel` has no user;
  `events` and `window` are `pub mod` with no user outside the directory.

## Decided in
`docs/runtime/2026-09-03-wayland-driver.md`; CLAUDE.md "Platform Notes".
