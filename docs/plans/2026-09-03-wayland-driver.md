# A Wayland display driver

**Goal: a second Linux driver, without giving up what makes `PlatformOps` worth
having.**

`PlatformOps` is the seam every display backend implements, and its value is
stated in its own doc comment:

> Outbound events are *returned* via `DriverOut` rather than sent, so an
> implementation can be driven and observed with no engine, no scheduler, and
> no channels in the loop.

That is a purity claim: messages in, `DriverOut` out, nothing retained. It is
what makes the X11 driver testable without an X server, and it is the property
this plan spends — deliberately, and only partly — to get a Wayland driver
built in reasonable time.

## The decision, and its price

**V1 uses `wayland-client`** (Smithay's bindings) rather than hand-bound FFI
against `libwayland-client`.

The price is exactly one thing. `wayland-client` delivers protocol events
through `Dispatch::event(state: &mut Self, …)` — the dispatch callback receives
`&mut State` and nothing else. `EventQueue<D>` and `QueueHandle<D>` are
`'static` in `D`, so `State` cannot borrow; there is no seam through which
`&mut DriverOut` reaches a callback. (`UserData` is per-object and also
`'static`, so it is not one either.)

So `WaylandOps` must **stage**: dispatch callbacks write into a buffer the ops
owns, and `handle_os` drains it into the caller's `out`. The driver retains
state between dispatch and drain. It is no longer pure.

What survives, and what does not:

- **Survives:** the property the trait doc actually names. The ops still hold
  no engine handle, still send nothing, are still drivable and observable with
  no scheduler and no channels. Every unit test that would have worked, works.
- **Survives:** zero per-frame allocation, by the same argument `DriverOut`
  already makes for its own `Vec` — reused, never shrunk, and an empty `Vec`
  never touches the heap.
- **Does not survive:** the driver as a function of its inputs alone. There is
  now a field whose contents depend on when dispatch last ran.

**Make the staging field a `DriverOut`.** Not a `Vec<DisplayEvent>` beside it:
the same type, so the staging API is the one the rest of the driver already
uses, and the drain is `out.emits.append(&mut self.staged.emits)` — a move of
the elements that leaves the source's capacity intact, so no allocation and no
second vocabulary. The impurity shrinks to "the ops owns a `DriverOut` between
dispatch and drain," which is the smallest honest statement of it.

One mechanical note found by building it: inside a `Dispatch` impl, a method
named `event` on the ops collides with `Dispatch::event`. Reaching the staging
buffer through its field (`self.staged.event(…)`) is both the fix and the
clearer spelling.

### What V2 would recover

The impurity is bounded and has a known exit. `wayland-client` sits on
`wayland-backend`, which exposes a lower-level `ObjectData` interface whose
callbacks are not tied to a `'static` dispatch state. If the staging field ever
costs more than it saves — a bug it hides, or a test it complicates — that is
the escape, and it does not require abandoning the crate. **Do not take it
pre-emptively:** hand-binding `xdg_shell`, `wl_seat` and the xkbcommon
keymap-over-fd path is hundreds of lines of `unsafe` to avoid one field, which
is the trade CLAUDE.md's "subtract before you add" exists to decline.

## What was measured, not assumed

All of the following was run in a container with **no GPU and no `/dev/dri`**,
against `weston --backend=headless`.

| question | result |
|---|---|
| `Send + 'static` for `Connection`, `EventQueue<S>`, and the state | all three satisfy it — `PlatformOps`' bound is met |
| connection fd reachable for a hand-rolled poll | yes, `prepare_read()` → `connection_fd()` |
| dispatch model | events land in `&mut State`; 19 collected in one registry roundtrip |
| **block, then wake from another thread** | blocked, interrupted at 300.26 ms by an `eventfd` write |
| full present path | `xdg_surface.configure`/`ack_configure`, `wl_shm` buffer, `wl_surface.commit`, frame callback, `wl_buffer.release` — all round-trip |

Dependency footprint: 22 crates, nearly all protocol codegen
(`wayland-scanner`, `quick-xml`) or already ubiquitous (`bitflags`, `rustix`,
`smallvec`).

The last row is the one that matters most, because it is the phase-1 spike:
`wl_shm` is a CPU-pixel path, which is what this renderer produces anyway, so
the absence of a GPU costs nothing. A dmabuf path would not be testable here —
and there isn't one on Linux to test.

## The mapping

| concern | X11 today | Wayland |
|---|---|---|
| present | `XCreateImage` + `XPutImage` + `XFlush` | `wl_shm` pool → `attach`/`damage`/`commit` |
| buffer ownership | implicit; `XPutImage` copies | **`wl_buffer.release`, asynchronous** |
| frame pacing | `XFlush`, no vsync source | `wl_callback` — a real one |
| window | Xlib window + WM hints | `xdg_surface` + `xdg_toplevel` |
| input | `XNextEvent` / XKB | `wl_seat`, xkbcommon keymap over an fd |
| resize | `ConfigureNotify` | **`configure` + `ack_configure` handshake** |

Two rows are new work rather than translation, and neither is attributable to
the library — raw FFI pays both identically.

### `Blitted` becomes an asynchronous emit

X11's `Present` is synchronous: `XPutImage` copies, so the buffer is reusable
immediately and `out.blitted(window)` fires inside `handle_data`. Wayland
forbids reuse until `wl_buffer.release` arrives, so `Blitted` must be emitted
from a *later* `handle_os`.

The architecture already permits this — `PlatformActor::flush` runs after every
handler, `handle_os` included, and `DriverEmit::Blitted` goes no further than
`WindowKeeper::rest`. What changes is timing: the keeper is bufferless for
longer than under X11, so **two buffers**, not one, or the engine stalls
waiting for a release.

The spike turned up an ordering fact that constrains the rotation design:

```
configure serial=1 acked
wl_buffer.release      <- arrived FIRST
frame callback (vsync)
```

`release` and `frame` are independent signals and `release` can precede the
callback. Rotation must not assume frame-then-release ordering, in either
direction.

### The `Idle` contract needs a hand-rolled poll

X11's `handle_os` blocks in `XNextEvent` when the scheduler reports
`SystemStatus::Idle`, and `X11Waker` interrupts it *in band* by posting a
`ClientMessage` into the same connection. Wayland has no in-band injection: you
cannot post an event into your own connection from another thread.

So the block is a `poll()` over `{connection fd, wake eventfd}`, using
`prepare_read()` rather than the library's `blocking_dispatch` — which polls
only the connection fd and therefore could not be woken. Measured above: it
blocks and a background thread interrupts it.

**One footgun, and it deserves a test.** If the poll returns because of the
wake fd rather than the connection fd, the `ReadEventsGuard` must be *dropped*,
not read — reading it leaves the connection mid-read. This is inherent to
Wayland's read protocol (`wl_display_prepare_read` / `cancel_read`), not to the
binding.

### The waker gets better

Worth stating because it is the one place Wayland is simpler. `X11Waker`
needs `XInitThreads`, an `unsafe impl Send` over a raw `*mut Display`, and a
`Mutex<Option<WakerInner>>` that is empty until the window exists — so a
`wake()` before then is silently dropped. `CocoaWaker` needs a hand-rolled
`WAKE_EVENT_QUEUED` token because the NSEvent queue does not coalesce.

A Wayland waker is a write to an `eventfd`. No unsafe `Send`, no
initialisation ordering hole, works before the surface exists, and the
counter coalesces for free.

## What CI forces

`scripts/check-driver-cfg-coverage.sh` is a hand-maintained ledger keyed on the
`use_*_display` cfgs a `build.rs` can select. Declaring `use_wayland_display`
**fails the build** until an entry records how CI covers it. That is the check
that caught the headless driver sitting 22-errors-broken behind a green history
for years, and it is working as designed here: this plan cannot land as code
alone, it has to land with a job.

The job is affordable, which the measurements above establish: `weston
--backend=headless` needs no GPU, and `weston` + `libwayland-dev` +
`wayland-protocols` are stock packages.

There is also a smaller structural change. `platform/mod.rs` selects
`ActivePlatform` by `target_os` alone:

```rust
#[cfg(target_os = "linux")]
pub type ActivePlatform = crate::display::platform::PlatformActor<linux::LinuxOps>;
```

while `build.rs` already emits `use_x11_display`. A second Linux driver forces
that selection onto the cfg — small, but it touches the X11 path, so it lands
on its own.

## Phasing

1. **Spike — done.** Recorded above: the whole present path round-trips under
   weston headless.
2. **The seam.** `display_wayland` feature, `use_wayland_display` in
   `build.rs`, `ActivePlatform` selected by cfg rather than `target_os`, the
   coverage-ledger entry, and a CI job that builds it and runs weston headless.
   Nothing renders yet; the gate is honest first.
3. **The driver.** `WaylandOps: PlatformOps` — staged `DriverOut`, shm pool,
   release-aware two-buffer rotation, frame-callback pacing, xdg_shell
   configure/ack.
4. **Input.** `wl_seat`, xkbcommon keymap over the fd, into core-term's key
   translation.
5. **A golden-frame test** in the job from step 2, so the driver is covered by
   execution and not only by compilation.

Steps 2 and 5 are the ones easiest to skip and the ones the repository's own
history argues hardest against skipping.
