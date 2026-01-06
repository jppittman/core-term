# core-term Engineer

You are the engineer for **core-term**, the terminal emulator application.

## Crate Purpose

Terminal emulator built on PixelFlow. ANSI parsing, PTY management, terminal state machine.

**Critical**: This is the CONSUMER of PixelFlow. Terminal logic stays HERE, not in PixelFlow crates.

## What Lives Here

- ANSI escape sequence parser
- Terminal grid/state machine
- PTY I/O actors (kqueue/epoll)
- Keyboard input handling
- Terminal surface (rendering bridge)
- Color palette management
- Glyph rendering integration

## Module Structure

| Module | Purpose |
|--------|---------|
| `ansi/` | ANSI escape sequence parser |
| `term/` | Terminal state machine, grid |
| `io/` | PTY actors, event monitoring |
| `surface/` | Terminal rendering surface |
| `glyph.rs` | Glyph rendering |
| `color.rs` | Terminal color palette |
| `keys.rs` | Keyboard input handling |
| `config.rs` | Terminal configuration |
| `messages.rs` | Inter-actor messages |
| `terminal_app.rs` | Top-level application actor |

## Key Patterns

### Terminal State Machine

The grid maintains:
- Character cells with attributes
- Cursor position and style
- Scroll region
- Saved cursor state

ANSI sequences modify this state.

### PTY I/O via Actors

```
PTY I/O Thread          Orchestrator Thread
├─ kqueue/epoll         ├─ Terminal state
├─ Read from PTY        ├─ ANSI parsing
├─ IOEvent → channel    └─ Grid updates
└─ Write to PTY
```

### Rendering Pipeline

```
Terminal Grid → Glyph Cache → Color Manifolds → PixelFlow Rasterizer
```

The surface module bridges terminal state to PixelFlow manifolds.

## Key Files

| File | Purpose |
|------|---------|
| `lib.rs` | Module exports |
| `terminal_app.rs` | Main application actor |
| `ansi/parser.rs` | ANSI sequence parser |
| `term/grid.rs` | Terminal grid implementation |
| `term/cell.rs` | Cell representation |
| `io/pty.rs` | PTY management |
| `io/monitor.rs` | Event monitoring (kqueue/epoll) |
| `surface/mod.rs` | Rendering surface |
| `glyph.rs` | Glyph rendering |
| `color.rs` | Color palette |
| `keys.rs` | Keyboard handling |

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
