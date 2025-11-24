# core-term Architecture

## Overview

core-term is a terminal emulator built on an actor-based concurrency model. The architecture is designed to solve the **PTY starvation problem**: when shell output floods the system, we need to ensure intermediate frames are rendered, not just the final state.

The system is **bidirectional**, not linear:
- **Input sources**: PTY (shell output), Display (UI events)
- **Output targets**: PTY (user keystrokes), Display (rendered frames)

## Design Principles

### Carmack Principle: No Empty Layers
> "What would Carmack do?"

We avoid architectural layers that don't perform real work. Each component has a clear responsibility and does actual computation, not just message routing.

### Errors Are Errors
Expected behavior is not an error. For example, `TryRecvError::Empty` when checking if the display is ready is normal flow control, not an error condition.

### No Raw Mutexes in 2024
Use channels for communication between threads. Mutexes should only appear in channel implementations, not application code.

## Thread Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Main Thread                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Platform (MacosPlatform)                                    │ │
│  │  - Owns CocoaDriver (macOS requirement)                     │ │
│  │  - Owns Renderer + SoftwareRasterizer                       │ │
│  │  - Rasterizes snapshots into framebuffer                    │ │
│  │  - Executes driver commands                                 │ │
│  │  - Sends Ready signal to Orchestrator                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Orchestrator Thread                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Orchestrator                                                │ │
│  │  - Owns TerminalEmulator (direct ownership, same thread)    │ │
│  │  - Drains events from all sources (PTY, Display, Vsync)     │ │
│  │  - Processes events through TerminalEmulator                │ │
│  │  - Generates snapshots ONLY when:                           │ │
│  │    1. Frame was requested (vsync or user input)             │ │
│  │    2. Display is ready (signaled via ready_rx)              │ │
│  │  - Routes actions to PTY/Display actors                     │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      PTY Thread                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ EventMonitorActor                                           │ │
│  │  - Owns NixPty                                              │ │
│  │  - Owns AnsiProcessor (parallel parsing)                    │ │
│  │  - Reads shell output from PTY                              │ │
│  │  - Parses ANSI sequences                                    │ │
│  │  - Sends IOEvent(Vec<AnsiCommand>) to Orchestrator          │ │
│  │  - Receives Write actions to send user input to shell       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Vsync Thread                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ VsyncActor                                                  │ │
│  │  - Wakes at target_fps (default: 60 FPS)                    │ │
│  │  - Sends RequestFrame to Orchestrator                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## The Backpressure Mechanism

The core innovation solving PTY starvation is a **two-channel handshake** between Orchestrator and Platform:

```rust
// Two 1-slot sync_channels for backpressure
let (snapshot_tx, snapshot_rx) = sync_channel(1);  // Orchestrator → Platform
let (ready_tx, ready_rx) = sync_channel(1);        // Platform → Orchestrator

// Initial state: Platform signals it's ready
ready_tx.send(SnapshotRequest::Ready)?;
```

### How It Works

1. **Event Draining**: Orchestrator drains all pending events with `try_recv()` to coalesce state updates
   ```rust
   loop {
       match event_rx.try_recv() {
           Ok(event) => process_event(event),
           Err(TryRecvError::Empty) => break,
           Err(TryRecvError::Disconnected) => return,
       }
   }
   ```

2. **Conditional Snapshot Generation**: Only generate snapshot when BOTH conditions are true:
   - A frame was requested (by vsync or user input)
   - Display is ready (check `ready_rx.try_recv()`)

   ```rust
   if frame_requested {
       match ready_rx.try_recv() {
           Ok(SnapshotRequest::Ready) => {
               let snapshot = term_emulator.get_render_snapshot()?;
               snapshot_tx.send(snapshot)?;
           }
           Err(TryRecvError::Empty) => {
               // Display still processing previous frame - skip
               // This is NOT an error, it's flow control
           }
           Err(TryRecvError::Disconnected) => return Err(...),
       }
   }
   ```

3. **Platform Processes and Signals**: Platform rasterizes, executes, then signals ready
   ```rust
   // Platform receives snapshot
   let snapshot = snapshot_rx.recv()?;

   // Rasterize into framebuffer
   let render_commands = renderer.prepare_render_commands(&snapshot, ...);
   compile_into_buffer(&mut rasterizer, render_commands, framebuffer, ...);

   // Execute driver commands
   driver.execute_driver_commands(driver_commands)?;

   // Signal ready for next snapshot
   ready_tx.send(SnapshotRequest::Ready)?;
   ```

### Why This Solves PTY Starvation

- **Event coalescing**: PTY flood gets drained and coalesced into single terminal state update
- **No wasted work**: Never generate snapshot if display is still busy with previous frame
- **Guaranteed progress**: Vsync RequestFrame events are processed even during PTY flood
- **No stale frames**: Only latest terminal state is rendered

## Orchestrator + Terminal: Same Thread

A key design decision: **Orchestrator owns TerminalEmulator via direct function calls**, not message passing.

### Rationale

- TerminalEmulator is a **pure state machine** with no I/O
- Doesn't benefit from separate thread
- Message passing overhead would be pure waste
- Logically separate but physically colocated
- Can split to separate thread later if profiling shows benefit

### Implementation

```rust
// Orchestrator thread owns Terminal directly
let mut term_emulator = TerminalEmulator::new(cols, rows);

// Process events through direct function call
let action = term_emulator.interpret_input(EmulatorInput::Ansi(command));
```

This is **not** a violation of actor principles - it's applying Carmack's principle of avoiding empty layers.

## Communication Pattern: Hub and Spoke

Orchestrator is the **central hub**. All actors communicate only with Orchestrator, never directly with each other.

```
    PTY Thread              Vsync Thread
        │                        │
        │ IOEvent           RequestFrame
        ├────────────┐      ┌────┘
        │            ▼      ▼
        │       ┌──────────────┐
        │       │ Orchestrator │
        │       └──────────────┘
        │            │      │
        │  Write     │      │ TerminalSnapshot
        └───────────┤      └────┐
                    ▼           ▼
              PTY Thread    Platform Thread
```

### Why Hub-and-Spoke?

- **Clear ownership**: Only one component owns each piece of state
- **Debuggability**: All cross-thread communication goes through one point
- **Backpressure**: Orchestrator can implement flow control
- **No distributed state**: Avoids complex synchronization

## Component Ownership

| Component | Owner | Thread | Reason |
|-----------|-------|--------|--------|
| TerminalEmulator | Orchestrator | Orchestrator | Pure state machine, no I/O |
| AnsiProcessor | PTY Actor | PTY | Parallel parsing, avoid Orchestrator bottleneck |
| Renderer | Platform | Main | Needs PlatformState, produces RenderCommands |
| SoftwareRasterizer | Platform | Main | Writes to framebuffer owned by CocoaDriver |
| CocoaDriver | Platform | Main | macOS requires UI on main thread |
| NixPty | PTY Actor | PTY | Blocking I/O operations |

### AnsiProcessor in PTY Thread

**Why?** Parsing ANSI sequences is CPU-intensive. If done in Orchestrator thread, it becomes a bottleneck.

**Solution:** PTY thread owns AnsiProcessor, parses bytes into `Vec<AnsiCommand>`, sends parsed commands to Orchestrator.

**Benefit:** Parallel parsing while Orchestrator processes previous frame.

### Renderer + Rasterizer on Platform Thread

**Why?** Framebuffer is owned by CocoaDriver on main thread. Copying framebuffer between threads would be expensive.

**Solution:** Platform owns both Renderer (snapshot → render commands) and Rasterizer (render commands → framebuffer pixels).

**Benefit:** Locality - snapshot → pixels happens in one thread, no copying.

## Data Flow

### User Types Key

```
1. Cocoa receives NSEvent
2. CocoaDriver translates to BackendEvent::Key
3. Platform sends PlatformEvent::BackendEvent to Orchestrator
4. Orchestrator processes through TerminalEmulator
5. TerminalEmulator returns EmulatorAction::WritePty(bytes)
6. Orchestrator sends PlatformAction::Write to PTY Actor
7. PTY Actor writes bytes to shell
```

### Shell Outputs Text

```
1. PTY Actor reads bytes from NixPty
2. PTY Actor parses with AnsiProcessor → Vec<AnsiCommand>
3. PTY Actor sends PlatformEvent::IOEvent to Orchestrator
4. Orchestrator drains all IOEvents, coalescing state updates
5. Orchestrator processes commands through TerminalEmulator
6. Vsync sends RequestFrame (sets frame_requested = true)
7. Orchestrator checks ready_rx.try_recv()
8. If Ok(Ready), generate snapshot and send to Platform
9. Platform rasterizes into framebuffer
10. Platform executes driver commands (presents frame)
11. Platform sends SnapshotRequest::Ready back to Orchestrator
```

## Shutdown Protocol

1. User closes window → CocoaDriver generates BackendEvent::CloseRequested
2. Platform sends to Orchestrator
3. Orchestrator sends PlatformAction::ShutdownComplete to Platform
4. Platform exits main loop
5. Drop handlers clean up threads

## Future Optimizations

- **Terminal as separate thread**: If profiling shows Orchestrator bottleneck, can move TerminalEmulator to separate actor
- **Render/Raster split**: If rendering is slow, could split Renderer to Orchestrator thread, keep Rasterizer on Platform
- **Multi-threaded rasterization**: Divide framebuffer into horizontal bands, rasterize in parallel

## References

- [STYLE.md](STYLE.md) - Code style conventions
- `src/orchestrator/` - Orchestrator implementation
- `src/platform/macos.rs` - macOS platform integration
- `src/term/` - Terminal state machine
