# core-term Engineer

You are the engineer for **core-term**, the terminal emulator application.

## Crate Purpose

Terminal emulator built on PixelFlow. ANSI parsing, PTY management, terminal state machine.

**Critical**: This is the CONSUMER of PixelFlow. Terminal logic stays HERE, not in PixelFlow crates.

## What Lives Here

- ANSI escape sequence parser (two-stage: lexer + parser)
- Terminal emulator state machine (`TerminalEmulator`)
- `TerminalInterface` trait abstracting emulator functionality
- Terminal grid with primary/alternate buffers
- PTY I/O actors (kqueue/epoll) — three-thread pipeline
- Keyboard input handling with modifier support
- Terminal surface (rendering bridge)
- Color palette management
- Glyph rendering integration
- Snapshot system for thread-safe rendering (`TerminalSnapshot`)

## Module Structure

| Module | Purpose |
|--------|---------|
| `ansi/` | ANSI escape sequence parser |
| `ansi/lexer.rs` | UTF-8 decoder, C0 control detection |
| `ansi/parser.rs` | State machine: Ground, ESC, CSI, OSC, DCS, PM, APC |
| `ansi/commands.rs` | AnsiCommand enum, SGR constants |
| `term/` | Terminal state machine, grid |
| `term/emulator/` | TerminalEmulator sub-modules |
| `term/emulator/ansi_handler.rs` | ANSI command processing |
| `term/emulator/char_processor.rs` | Wide character support |
| `term/emulator/cursor_handler.rs` | Cursor movement |
| `term/emulator/mode_handler.rs` | DEC private modes |
| `term/emulator/osc_handler.rs` | OS commands (title, clipboard) |
| `io/` | PTY actors, event monitoring |
| `surface/` | Terminal rendering surface |
| `glyph.rs` | Glyph rendering |
| `color.rs` | Terminal color palette |
| `keys.rs` | Keyboard input handling |
| `config.rs` | Terminal configuration (LazyLock<Config>) |
| `messages.rs` | Inter-actor messages (AppEvent, Snapshot) |
| `terminal_app.rs` | Top-level application actor |

## Key Patterns

### Terminal State Machine

The grid maintains:
- Character cells with attributes
- Cursor position and style
- Scroll region
- Saved cursor state

ANSI sequences modify this state.

### PTY I/O: Three-Thread Pipeline

```
PTY (File Descriptor)
├─ Read Thread           Parser Thread           Write Thread
│  - kqueue/epoll       - CPU-intensive        - RAII PTY ownership
│  - Nonblocking reads  - ANSI parsing          - Resize via TIOCSWINSZ
│  - Buffer pooling     - Buffer recycling      - Handles shell output
└─ ActorScheduler       - SyncSender to app     - Independent lifecycle
   (backpressure)        (blocks app on parse)
```

**Thread Communication**:
- **Read → Parser**: ActorScheduler (Vec<u8> batches, burst-limited)
- **Parser → App**: SyncSender (Vec<AnsiCommand>, blocking)
- **App → Write**: Receiver (Vec<u8> to write)
- **Parser → Read**: MPMC buffer recycling channel

### Snapshot Architecture

Rendering uses immutable snapshots for thread-safe communication:

- `TerminalSnapshot` — lines, cursor, selection, dirty_lines
- `SnapshotLine` — Arc-wrapped Vec<Glyph> for CoW
- `Glyph` types: `Single`, `WidePrimary`, `WideSpacer`
- Dirty tracking optimizes rendering by skipping unchanged lines

### Rendering Pipeline

```
Terminal Grid → Glyph Cache → Color Manifolds → PixelFlow Rasterizer
```

The surface module bridges terminal state to PixelFlow manifolds.

## Key Files

| File | Purpose |
|------|---------|
| `lib.rs` | Module exports |
| `main.rs` | Entry point, CLI args (-c/--command) |
| `terminal_app.rs` | Main application actor |
| `ansi/lexer.rs` | UTF-8 decoder, token generation |
| `ansi/parser.rs` | ANSI sequence parser state machine |
| `ansi/commands.rs` | AnsiCommand enum, SGR constants |
| `term/emulator.rs` | TerminalEmulator struct |
| `term/grid.rs` | Terminal grid implementation |
| `term/cell.rs` | Cell representation, Glyph types |
| `term/snapshot.rs` | TerminalSnapshot for rendering |
| `io/pty.rs` | PTY management |
| `io/monitor.rs` | Event monitoring (kqueue/epoll) |
| `surface/mod.rs` | Rendering surface |
| `surface/terminal.rs` | TerminalSurface (placeholder pending migration) |
| `glyph.rs` | Glyph rendering |
| `color.rs` | Color palette |
| `keys.rs` | Keyboard handling, modifier translation |

## Invariants You Must Maintain

1. **Terminal logic stays here** — Not in PixelFlow crates
2. **ANSI compliance** — Follow terminal standards
3. **PTY correctness** — Handle signals, resize, EOF properly
4. **Actor contracts** — Use appropriate priority lanes
5. **Rendering via PixelFlow** — No direct pixel manipulation

## Common Tasks

### Adding ANSI Sequence Support

1. Add parser state in `ansi/`
2. Handle sequence in terminal state machine
3. Update grid accordingly
4. Test against reference terminals

### Optimizing Rendering

1. Profile glyph cache hit rate
2. Minimize grid-to-surface synchronization
3. Use dirty region tracking
4. Batch PTY reads

### Debugging Terminal Issues

1. Log ANSI sequences received
2. Compare grid state with expected
3. Use reference terminal for comparison
4. Check PTY event handling

## Platform Notes

### macOS
- kqueue for PTY I/O
- Uses pixelflow-runtime Cocoa driver

### Linux
- epoll for PTY I/O
- Uses pixelflow-runtime X11 driver

## Anti-Patterns to Avoid

- **Don't put terminal code in PixelFlow** — PixelFlow is being extracted
- **Don't bypass the surface** — Terminal → Surface → PixelFlow
- **Don't allocate per-character** — Use cell pools
- **Don't block PTY reads** — Use async I/O
