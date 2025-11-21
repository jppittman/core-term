# Actor-Based Threading Architecture

## Executive Summary

core-term uses a **three-thread actor model** with message-passing channels to separate concerns cleanly:

1. **Display Driver Thread** (main thread) - Platform-specific UI event loops
2. **Orchestrator Thread** - Platform-agnostic application logic
3. **PTY I/O Thread** - Asynchronous pseudo-terminal I/O

This architecture provides zero-latency input handling, clean separation of concerns, and platform symmetry.

## Problem Statement

### Initial Constraint: macOS Requirements

**Hard requirement from Apple:** AppKit/Cocoa MUST run on the main thread.

### Initial Approach: Orchestrator-Owned Event Loop

We initially tried to implement an event-driven architecture where the Orchestrator owned the main event loop:

```rust
// Orchestrator owns EventMonitor (kqueue/epoll)
impl Orchestrator {
    fn run_event_loop() {
        let event_monitor = EventMonitor::new();
        event_monitor.add(wake_fd, EventFlags::READ);

        loop {
            event_monitor.events(&mut buffer, 16ms_timeout);
            self.process_event_cycle();
        }
    }
}
```

**Why we tried this:**
- Single select/epoll/kqueue in one place
- Orchestrator controls timing and polling
- Seemed to provide unified control flow

**Problems discovered:**

1. **Raw FD exposure**: Had to expose `wake_fd: RawFd` through Platform abstraction, breaking type safety
2. **Unsafe required**: Reading from raw FD requires `unsafe { BorrowedFd::borrow_raw(fd) }`
3. **16ms polling latency**: Cocoa has no exposed event FD, so we'd need to poll every 16ms for UI events
   - This means **16ms worst-case latency on keypress** - unacceptable!
4. **Platform abstraction violation**: Orchestrator needed platform-specific EventMonitor knowledge
5. **No platform symmetry**: Linux could use true select(), macOS had to fake it with polling

### The Realization: Platform Should Own the Loop

Key insight from design discussion:

> "I'd rather let cocoa own the main loop than have 16 ms latency on a key press."

**The fundamental issue:** We were trying to impose a unified event loop model on platforms with different requirements:
- macOS: `NSApp.run()` blocks and owns the thread - this is the ONLY way to get zero-latency Cocoa events
- Linux: epoll/select can block on multiple FDs naturally

Rather than force macOS into Linux's model with polling hacks, we should embrace platform-specific event loops and provide a **unified abstraction at the message-passing level**, not the event loop level.

## Architecture Design

### Three-Thread Actor Model

```
┌─────────────────────────────────────────────────────────────┐
│                       MAIN THREAD                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Display Driver Actor (Cocoa/X11)                      │ │
│  │                                                        │ │
│  │  macOS: NSApp.run() - native event loop, zero latency │ │
│  │  Linux: X11 event loop - can run on any thread        │ │
│  │                                                        │ │
│  │  Responsibilities:                                     │ │
│  │  - Run platform-native event loop                     │ │
│  │  - Translate native events → PlatformEvent            │ │
│  │  - Send to Orchestrator via channel                   │ │
│  │  - Receive PlatformAction from Orchestrator           │ │
│  │  - Execute rendering/windowing commands               │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                               ▲ │
                    UI Events  │ │ Render/UI Commands
                  (BackendEvent)│ │ (PlatformAction)
                               │ ▼
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR THREAD                       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Orchestrator Actor                                    │ │
│  │                                                        │ │
│  │  Event Loop: SELECT on multiple channels              │ │
│  │  - Display events channel (BackendEvent)              │ │
│  │  - PTY events channel (IOEvent)                       │ │
│  │                                                        │ │
│  │  Owns State:                                           │ │
│  │  - TerminalEmulator                                    │ │
│  │  - AnsiProcessor                                       │ │
│  │  - Renderer                                            │ │
│  │                                                        │ │
│  │  Responsibilities:                                     │ │
│  │  - Process PTY data (ANSI parsing, terminal updates)  │ │
│  │  - Process UI events (keyboard, mouse, resize)        │ │
│  │  - Generate render commands                           │ │
│  │  - Platform-agnostic logic                            │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                               ▲ │
                    PTY Data   │ │ PTY Commands
                    (IOEvent)  │ │ (Write/Resize)
                               │ ▼
┌─────────────────────────────────────────────────────────────┐
│                      PTY I/O THREAD                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  EventMonitorActor                                     │ │
│  │                                                        │ │
│  │  Event Loop: kqueue/epoll on PTY file descriptor      │ │
│  │                                                        │ │
│  │  Responsibilities:                                     │ │
│  │  - Poll PTY fd for readable events                    │ │
│  │  - Read PTY output → send to Orchestrator             │ │
│  │  - Receive write commands from Orchestrator           │ │
│  │  - Write to PTY, handle resize                        │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Message Types

We reuse existing message types from the codebase:

**Display → Orchestrator:**
```rust
PlatformEvent::BackendEvent(BackendEvent) where BackendEvent is:
  - Key { symbol, modifiers, text }
  - MouseButtonPress/Release/Move
  - Resize { width_px, height_px }
  - FocusGained/FocusLost
  - PasteData { text }
  - CloseRequested
```

**PTY → Orchestrator:**
```rust
PlatformEvent::IOEvent { data: Vec<u8> }  // Shell output
```

**Orchestrator → Display:**
```rust
PlatformAction:
  - Render(Vec<RenderCommand>)
  - SetTitle(String)
  - RingBell
  - CopyToClipboard(String)
  - SetCursorVisibility(bool)
  - RequestPaste
```

**Orchestrator → PTY:**
```rust
PlatformAction:
  - Write(Vec<u8>)
  - ResizePty { cols, rows }
```

### Control Flow

#### Startup Sequence

```rust
fn main() -> Result<()> {
    // 1. Platform creates all actors
    let platform = MacosPlatform::new(cols, rows, shell, args)?;

    // 2. Platform::run() blocks in native event loop
    platform.run()?;

    Ok(())
}
```

**Inside Platform::new():**
1. Spawn EventMonitorActor (PTY thread) - get channels
2. Spawn OrchestratorActor (logic thread) - get channels
3. Connect channels: PTY ↔ Orchestrator ↔ Display
4. Initialize native display (CocoaDriver, X11Driver)
5. Return platform ready to run

**Inside Platform::run():**
- macOS: `NSApp.run()` blocks forever, callbacks forward events via channel
- Linux: Event loop blocks on X11 connection, forwards events via channel

#### Event Flow Example: Keypress

```
1. User presses 'a'
2. Cocoa generates NSEvent
3. NSApp.run() delivers to delegate callback (ZERO LATENCY!)
4. CocoaDriver::handle_key_event()
   → BackendEvent::Key { symbol, modifiers, text }
   → display_to_orchestrator_tx.send()

5. Orchestrator thread wakes from select()
   → Receives BackendEvent::Key
   → Processes: term.interpret_input(key)
   → Generates: EmulatorAction::WritePty("a".bytes())
   → orchestrator_to_pty_tx.send(Write("a"))

6. PTY thread wakes from channel recv
   → Writes "a" to PTY fd
   → Shell processes keystroke

7. Shell outputs response
8. PTY fd becomes readable
9. EventMonitorActor wakes from kqueue
   → Reads PTY data
   → pty_to_orchestrator_tx.send(IOEvent { data })

10. Orchestrator processes shell output
    → ANSI parsing
    → Terminal state update
    → Generate render commands
    → orchestrator_to_display_tx.send(Render(commands))

11. Display thread receives render commands
    → Executes Metal/OpenGL drawing
    → Screen updates
```

## Design Rationale

### Why Three Threads?

**Thread 1 (Display - Main):**
- **Must be main thread on macOS** (Apple requirement)
- Runs platform-native event loop (blocking)
- Zero-latency input handling
- Isolates platform-specific display code

**Thread 2 (Orchestrator - Background):**
- **Must NOT be main thread** - we want it to block on a unified event channel
- Platform-agnostic: identical logic on macOS/Linux/Windows
- Owns all application state (terminal, renderer)
- Clean actor with defined inputs/outputs

**Thread 3 (PTY I/O - Background):**
- Keeps PTY I/O non-blocking and async
- Isolates kqueue/epoll platform differences
- Future: easy migration to io_uring on Linux

### Why Orchestrator in Separate Thread?

**Key insight:** We want the Orchestrator to block on events from multiple sources (Display and PTY).

**Solution: Unified Event Channel**

Instead of using external libraries with `select!` macros, we use a single `mpsc::channel` with an enum:

```rust
// Unified event type
enum OrchestratorEvent {
    Display(BackendEvent),
    Pty(Vec<u8>),
}

// Single channel - both actors send here
let (event_tx, event_rx) = mpsc::channel();

// Display thread clones sender
let display_tx = event_tx.clone();

// PTY thread clones sender
let pty_tx = event_tx.clone();

// Orchestrator blocks on ONE channel
loop {
    let event = event_rx.recv()?;
    match event {
        OrchestratorEvent::Display(backend_event) => {
            // Handle UI events (key, mouse, resize, etc.)
        }
        OrchestratorEvent::Pty(data) => {
            // Handle PTY data (ANSI parsing, terminal update)
        }
    }
}
```

**Benefits:**
- Zero external dependencies (just std::sync::mpsc)
- True blocking - wakes immediately when ANY event arrives
- Simple and clean
- Equivalent to `select()` but at the channel level

**If Orchestrator were on main thread:**
- macOS: `NSApp.run()` blocks the thread - we can't select() on anything else!
- We'd have to poll channels from Cocoa callbacks (tight coupling)
- Loses the clean actor model

**By moving Orchestrator to its own thread:**
- True select/crossbeam-channel multi-wait
- Platform-agnostic: same code on all platforms
- Display thread just forwards events, doesn't need to know about PTY

### Why Display Owns Main Thread?

**Alternative considered:** Put Orchestrator on main, Display on background thread

**Why rejected:**
1. **macOS requirement violated**: Cocoa must be on main thread
2. **Linux could work but asymmetric**: X11 doesn't care about thread
3. **But no benefit**: We still need channels for PTY, so why not use channels for Display too?

**By letting Display own main:**
- Satisfies platform requirements (Cocoa on main)
- Platform symmetry: both macOS and Linux use same channel-based design
- Orchestrator is identical on all platforms

### Why Not Just Two Threads?

**Alternative considered:** Orchestrator on main, PTY on background

**Why rejected:**
1. Orchestrator can't block on `select()` if macOS NSApp.run() owns the thread
2. Would need polling or callbacks from Cocoa (adds latency or complexity)
3. Loses platform symmetry

**Alternative considered:** Display + Orchestrator on main, PTY on background

**Why rejected:**
1. Tight coupling between Display and Orchestrator
2. Platform-specific code mixed with logic
3. Can't use select() on main thread (NSApp.run() blocks)

## Trade-offs

### Pros

✅ **Zero-latency input**: Native event loops provide immediate event delivery
✅ **Platform symmetry**: Same actor model on macOS/Linux/Windows
✅ **Clean separation**: Display/Logic/I/O completely isolated
✅ **Type safety**: Channels carry proper Rust types, no raw FDs in business logic
✅ **No unsafe code**: Channel-based communication is safe
✅ **Testable**: Orchestrator is pure logic, can be tested with mock channels
✅ **Future-proof**: Easy to add more actors (clipboard, ime, etc.)

### Cons

❌ **Channel overhead**: Small latency from mpsc/crossbeam channels (~microseconds)
❌ **More threads**: 3 threads vs 1 (but threads are cheap, and we need the separation)
❌ **Slightly more complex**: More moving parts than single-threaded

### Performance Considerations

**Channel latency:** Modern Rust channels (crossbeam/flume) have ~1-2μs overhead
- Negligible compared to PTY/display I/O (~1-10ms)
- User will never notice microsecond delays

**Memory:** Each thread has its own stack (~2MB default)
- 3 threads = ~6MB overhead
- Acceptable for a GUI terminal application

**CPU:** No busy-waiting or polling
- All threads block on events (select/recv/epoll)
- Minimal CPU usage when idle

## Alternatives Rejected

### Alternative 1: Orchestrator Owns Main Loop with Polling

**Approach:**
```rust
impl Orchestrator {
    fn run() {
        let monitor = EventMonitor::new();
        monitor.add(wake_fd, READ);
        loop {
            monitor.events(16ms_timeout);  // Poll every 16ms for Cocoa
            platform.poll_cocoa();  // Non-blocking
            process_events();
        }
    }
}
```

**Why rejected:**
- 16ms worst-case input latency (unacceptable)
- Requires exposing raw FDs through Platform abstraction
- Platform-specific code leaks into Orchestrator
- Polling wastes CPU

### Alternative 2: Orchestrator Owns Main Loop with Wake FD

**Approach:**
```rust
impl Orchestrator {
    fn run() {
        let monitor = EventMonitor::new();
        monitor.add(wake_fd, READ);  // Wake when PTY has data
        loop {
            monitor.events(NO_TIMEOUT);  // Block forever
            // But how do we wake for Cocoa events???
        }
    }
}
```

**Why rejected:**
- Cocoa doesn't expose an event FD to select() on
- Would need to poll Cocoa periodically (back to 16ms latency)
- Platform abstraction violated (exposing wake_fd)

### Alternative 3: Callbacks from Platform to Orchestrator

**Approach:**
```rust
impl CocoaDelegate {
    fn keyDown(event: NSEvent) {
        orchestrator.handle_key(event);  // Direct call
    }
}
```

**Why rejected:**
- Tight coupling between Platform and Orchestrator
- Orchestrator must be Send + Sync (threading complexity)
- Loses actor model benefits
- Hard to test (mocking becomes difficult)

### Alternative 4: Single-Threaded with Async/Tokio

**Approach:**
```rust
#[tokio::main]
async fn main() {
    tokio::select! {
        event = pty.read() => { },
        event = platform.poll() => { },
    }
}
```

**Why rejected:**
- NSApp.run() is blocking and can't be made async
- Would need to run Cocoa in separate thread anyway (back to threading model)
- Async adds complexity without benefits for this use case
- Terminal emulation is CPU-bound, not I/O-bound (async doesn't help)

## Implementation Notes

### Channel Design: Unified Event Channel

Use **std::sync::mpsc** with a unified event enum:
- No external dependencies
- Single channel for all events
- Both Display and PTY threads clone the sender
- Orchestrator blocks on single receiver
- Equivalent to `select()` but simpler

### Platform::run() Signature

```rust
trait Platform {
    fn run(self) -> Result<()>;  // Consumes self, blocks forever
}
```

Takes `self` (not `&mut self`) because:
- Platform owns the channels and actors
- `run()` never returns normally (only on error/shutdown)
- Consuming prevents misuse

### Graceful Shutdown

1. User closes window → Display sends `BackendEvent::CloseRequested`
2. Orchestrator receives close event
3. Orchestrator sends shutdown message to PTY thread
4. Orchestrator exits its loop
5. Display thread detects Orchestrator channel closed
6. Display thread exits its loop → Platform::run() returns

## Future Enhancements

### io_uring on Linux

Replace EventMonitorActor with io_uring-based implementation:
- Same actor interface
- Same message types
- Better performance (kernel-side polling)
- Zero-copy where possible

### Additional Actors

Easy to add more actors for:
- **Clipboard management** (system clipboard monitoring)
- **IME handling** (input method editor for CJK)
- **Tab management** (multiple terminal sessions)
- **SSH integration** (remote connections)

Each actor follows the same pattern: spawned thread, channels for communication.

## Conclusion

The three-thread actor model provides the best balance of:
- **Correctness**: Satisfies platform requirements (Cocoa on main)
- **Performance**: Zero-latency input handling
- **Maintainability**: Clean separation, testable components
- **Portability**: Platform-symmetric design

By embracing platform-specific event loops and providing abstraction at the message-passing level (not the event loop level), we achieve both native platform integration and clean cross-platform architecture.
