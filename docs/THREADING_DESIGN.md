# Threading Architecture Design

## Overview

This document describes the threading model for core-term, which separates platform-specific UI event loops from PTY I/O handling.

## Problem Statement

**macOS Constraint**: AppKit/Cocoa MUST run on the main thread. This is a hard requirement from Apple.

**Current Issue**: We're trying to poll Cocoa events via `nextEventMatchingMask` with `untilDate:nil`, but Cocoa windows don't display properly without running the full `NSApplication.run()` event loop.

**Solution**: Separate concerns into dedicated threads with clear communication channels.

## Architecture

### Thread Model

```
┌─────────────────────────────────────────────────────────────┐
│                       MAIN THREAD                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Platform Event Loop (Cocoa/X11/Wayland)               │ │
│  │  - NSApp.run() on macOS (blocking, owns thread)        │ │
│  │  - X11 event processing on Linux                       │ │
│  │  - Owns window/rendering                               │ │
│  └────────────────────────────────────────────────────────┘ │
│                            ▲ │                              │
│                   Commands │ │ Events                       │
│                            │ ▼                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Orchestrator                                          │ │
│  │  - Owns all state (term, renderer, ansi parser)       │ │
│  │  - Coordinates message passing                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                            ▲ │                              │
│                      Data  │ │ Write/Resize                 │
│                            │ ▼                              │
└─────────────────────────────────────────────────────────────┘
                               │
                    mpsc channel│
                               │
┌─────────────────────────────────────────────────────────────┐
│                    BACKGROUND THREAD                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  PTY I/O Thread                                        │ │
│  │  - EventMonitor (kqueue/epoll/io_uring)                │ │
│  │  - Polls PTY file descriptor                           │ │
│  │  - Reads PTY data → sends to main thread               │ │
│  │  - Receives write commands from main thread            │ │
│  └────────────────────────────────────────────────────────┘ │
│                            ▲ │                              │
│                            │ │ (direct syscalls)            │
│                            │ ▼                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  PTY (File Descriptor)                                 │ │
│  │  - Connected to shell process                          │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Channel Communication

**Uses existing types from `platform/actions.rs` and `platform/mod.rs`:**

**PTY Thread → Main Thread:**
```rust
PlatformEvent::IOEvent { data: Vec<u8> }  // Shell output
PlatformEvent::BackendEvent(BackendEvent::CloseRequested)  // PTY closed
```

**Main Thread → PTY Thread:**
```rust
PlatformAction::Write(Vec<u8>)              // Write to shell
PlatformAction::ResizePty { cols, rows }    // Resize PTY
```

**Note:** We reuse existing message types - no new types needed! The architecture was already designed for this.

## Implementation Plan

### Phase 1: Thread Separation (Current PR)

1. **Create PTY I/O thread** in `MacosPlatform::new()`
   - Spawn background thread
   - Move EventMonitor + PTY to background thread
   - Set up bidirectional mpsc channels

2. **Update CocoaDriver** to run native event loop
   - Remove manual event polling
   - Let `NSApp.run()` own the main thread
   - Process events via delegate callbacks

3. **Update `MacosPlatform::poll_events()`**
   - Check channel for PTY messages (non-blocking)
   - Poll Cocoa for UI events (already on main thread)
   - Combine and return all events

### Phase 2: Linux Refactor (Future)

Apply same pattern to Linux:
- Main thread: X11/Wayland event loop
- Background thread: epoll → PTY I/O
- Same channel-based communication

### Phase 3: io_uring Migration (Future)

Replace epoll with io_uring on Linux:
- Same threading model
- Better performance (kernel-side polling)
- Zero-copy I/O where possible

## Platform-Specific Notes

### macOS (Current Implementation)

**Main Thread:**
- `NSApplication.run()` - blocking, owns thread
- Cocoa delegate receives events
- Forwards to orchestrator via callback

**PTY Thread:**
- `kqueue` polls PTY fd
- Sends data via channel
- Receives write commands

### Linux (Current Implementation)

**Current:** Single-threaded with epoll

**Future:** Same pattern as macOS
- Main: X11/Wayland event loop
- Background: epoll + PTY
- Channels for communication

## Benefits

1. **Correctness**: Cocoa runs properly on main thread
2. **Separation**: UI and I/O concerns cleanly separated
3. **Performance**: Non-blocking I/O on background thread
4. **Future-proof**: Easy to migrate to io_uring
5. **Maintainability**: Clear ownership and communication patterns

## Trade-offs

**Pros:**
- Clean architecture
- Platform requirements satisfied
- Scalable to more complex scenarios

**Cons:**
- Channel overhead (minimal)
- Slightly more complex than single-threaded
- Need to ensure thread safety

## Open Questions

1. Should EventMonitor be `Send + Sync`?
   - **Decision:** Only accessed from PTY thread, doesn't need to be Sync

2. How do we handle PTY thread panics?
   - **Decision:** Panic will propagate, main thread detects via channel closure

3. Should we use bounded or unbounded channels?
   - **Decision:** Bounded (e.g., 100 messages) to apply backpressure

## Success Criteria

- ✅ Window appears on screen (macOS)
- ✅ PTY data flows correctly
- ✅ Keyboard input works
- ✅ No blocking on either thread
- ✅ Clean shutdown
