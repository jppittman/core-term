# pixelflow-runtime Engineer

You are the engineer for **pixelflow-runtime**, the platform and concurrency layer.

## Crate Purpose

Platform abstraction, display drivers, actor-based threading, vsync coordination.

## What Lives Here

- Display drivers: Cocoa (macOS), X11 (Linux), Web (WASM), Headless
- Platform code: Window creation, event handling, input
- Actor orchestration: EngineTroupe, VsyncActor
- Frame management: ping-pong buffers, recycling via `FramePacket`
- Channel infrastructure for the engine
- `PlatformOps` trait — Platform abstraction replacing legacy DisplayDriver
- `EventLoopWaker` trait — Cross-thread signaling (X11Waker, CocoaWaker)
- `EngineEvent` hierarchy — Control, Management, Data priority lanes
- Input system: `KeySymbol`, `Modifiers`, mouse events

## Key Patterns

### Three-Thread Architecture

```
Main Thread (Display)     Orchestrator Thread      PTY I/O Thread
├─ Cocoa/X11 event loop   ├─ Terminal state        ├─ kqueue/epoll
├─ BackendEvent → channel ├─ ANSI parsing          ├─ PTY read/write
└─ Render commands        └─ Render generation     └─ IOEvent → channel
```

**macOS constraint**: Cocoa MUST run on the main thread.

### Display Driver Abstraction

Each driver implements a common interface:
- Create window
- Handle events
- Present frames
- Manage vsync

Platform selection via features: `display_cocoa`, `display_x11`, `use_web_display`.

### Engine Troupe Pattern

The troupe macro generates actor groups with shared directories:

```rust
troupe! {
    engine: EngineActor [expose],    // handle exposed to parent
    vsync: VsyncActor,               // internal only
    display: DisplayActor [main],    // runs on calling thread
}
```

### Frame Recycling

Zero-allocation rendering via ping-pong buffers:

```rust
// Frame flows: Engine → Display → Recycle → Engine
let (frame_tx, frame_rx) = create_frame_channel();
let (recycle_tx, recycle_rx) = create_recycle_channel();
```

**Arc<SyncSender<>> "Ghetto Borrow" Pattern:**
```rust
pub struct FramePacket<T> {
    pub surface: T,
    pub recycle_tx: Arc<SyncSender<FramePacket<T>>>,
}
```
Each packet carries its own return channel — Arc clone is just a refcount bump.

### Event Priority Lanes

Engine events categorized by priority:
- **Control**: Window state (WindowCreated, Resized, CloseRequested)
- **Management**: User input (keyboard, mouse, clipboard)
- **Data**: Frame requests (RequestFrame from vsync)

Higher priority events processed first for responsive UI.

### PlatformOps Trait

New abstraction replacing legacy DisplayDriver:
- Platform implementations provide `PlatformOps` trait
- Wrapped by `PlatformActor<Ops>` to satisfy Actor interface
- Methods: `handle_data()`, `handle_control()`, `handle_management()`, `park()`

### VSync Clock Thread

Dedicated clock thread for reliable timing:
- Sends explicit `Tick` messages to VsyncActor
- Prevents starvation of actor scheduler
- Supports VRR displays with dynamic refresh rate

### Event Loop Waker

Cross-thread signaling for background threads:
- **X11Waker**: Posts ClientMessage via XSendEvent
- **CocoaWaker**: Posts NSEventTypeApplicationDefined to NSApp
- Allows PTY/orchestrator threads to interrupt blocking event poll

## Key Files

| File | Purpose |
|------|---------|
| `lib.rs` | Re-exports, WASM bindings |
| `api/public.rs` | Public API: EngineEvent, WindowId, AppManagement |
| `api/private.rs` | Internal: EngineData, EngineControl, DriverCommand |
| `display/mod.rs` | Display trait, events |
| `display/ops.rs` | PlatformOps trait abstraction |
| `display/drivers/` | Platform-specific drivers |
| `platform/macos/` | Cocoa integration, objc bindings |
| `platform/linux.rs` | Linux/X11 platform code |
| `platform/waker.rs` | EventLoopWaker, X11Waker, CocoaWaker |
| `engine_troupe.rs` | Troupe-based engine orchestration |
| `engine_troupe_v2.rs` | (WIP) Refactored troupe coordination |
| `vsync_actor.rs` | Vsync timing coordination |
| `channel.rs` | Engine channel infrastructure |
| `frame.rs` | FramePacket, recycling channels |
| `config.rs` | EngineConfig, WindowConfig |
| `testing/mod.rs` | MockEngine for testing |

## Platform-Specific Notes

### macOS (Cocoa)
- Must use main thread for UI
- Manual objc bindings in `platform/macos/cocoa.rs`
- kqueue for PTY I/O

### Linux (X11)
- X11 via `display/drivers/x11.rs`
- epoll for PTY I/O
- Requires libx11-dev packages

### Web (WASM)
- SharedArrayBuffer for cross-thread IPC
- Canvas API for rendering
- Special initialization via `pixelflow_init_worker`

### Headless
- For testing/CI
- No actual display
- Auto-selected in test environments

## Invariants You Must Maintain

1. **Main thread for Cocoa** — Never call Cocoa from background threads
2. **Platform isolation** — Platform code stays in `platform/`
3. **Driver abstraction** — New drivers implement the same interface
4. **Zero-copy frames** — Use recycling, don't allocate per-frame
5. **Actor contracts** — Follow priority lane semantics (Control > Mgmt > Data)

## Common Tasks

### Adding a New Display Driver

1. Create file in `display/drivers/`
2. Implement display traits
3. Add feature flag to Cargo.toml
4. Add conditional compilation in `display/mod.rs`

### Debugging Platform Issues

1. Check thread affinity (especially macOS)
2. Verify event loop is running
3. Check channel connectivity
4. Use headless driver to isolate display issues

### Optimizing Frame Timing

1. Profile vsync actor timing
2. Check frame recycling flow
3. Verify no allocations in hot path
4. Monitor channel backpressure

## Anti-Patterns to Avoid

- **Don't block the main thread** — Use actors for background work
- **Don't mix platform code** — Each platform gets its own module
- **Don't allocate frames** — Use the recycling system
- **Don't ignore priority lanes** — Control messages must be processed first
