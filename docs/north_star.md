# myterm Design Document - Target Architecture

**Version:** 1.4
**Date:** 2025-05-05
**Status:** Active

## 1. Introduction & Vision

`myterm` aims to be a correct, reasonably performant, and maintainable terminal emulator written in Rust. This document outlines a target architecture refactored from the initial `st`-inspired design towards a clearer separation of concerns, adopting an **Interpreter/Virtual Machine (VM) model** for its core logic.

The central idea is that the terminal's state machine (`TerminalEmulator`) acts as a self-contained VM. It processes a defined set of inputs (`EmulatorInput`) derived from PTY output and user actions, updates its internal state based on these inputs, and signals required external side-effects (`EmulatorAction`) without performing direct I/O. This approach enhances modularity, testability, and clarifies the role of each component.

A core principle is **simplicity and stability**, favoring well-established, CPU-based rendering techniques over GPU acceleration to avoid complex driver dependencies. Following the suckless philosophy, features like scrollback are considered **non-goals**, encouraging composition with tools like `tmux`.

## 2. Core Concepts & Terminology

* **Terminal Emulator (Interpreter/VM):** The core state machine (`struct TerminalEmulator`). Maintains the terminal grid, cursor, attributes, modes, etc. Interprets `EmulatorInput` to modify state and signals `EmulatorAction`s. **Does not perform I/O or rendering.** Contains dirty flags for rendering optimization.

* **Emulator Input (Instruction Set):** `enum EmulatorInput` wrapping `AnsiCommand` (from PTY) or `BackendEvent` (from user/platform).

* **Parser Pipeline:** (`AnsiLexer`, `AnsiParser`) Converts PTY byte stream -> `Vec<AnsiCommand>`. Runs outside the `TerminalEmulator`.

* **Emulator Action (Side-Effect Request):** `enum EmulatorAction` returned by `TerminalEmulator` (e.g., `WritePty`, `SetTitle`, `RingBell`) for the `Orchestrator` to execute.

* **Orchestrator (`main`):** Central coordinator. Runs event loop, manages PTY I/O, drives parser, feeds `EmulatorInput` to `TerminalEmulator`, executes `EmulatorAction`s, triggers `Renderer`.

* **Rendering Driver (`Driver` Trait):** Interface implemented by backend-specific components (`XDriver`, `ConsoleDriver`). Responsibilities:
    * Window/display setup and management.
    * Platform event handling and translation to `BackendEvent`.
    * **Provides concrete implementations for abstract drawing primitives** (e.g., `draw_text_run`, `fill_rect`, `present`).
    * Provides necessary info like event FD, font metrics.

* **Renderer (`Renderer` Struct/Module):** **Backend-agnostic** component responsible for:
    * **Querying the `TerminalEmulator` for dirty state** (e.g., calling `term.take_dirty_lines()`).
    * Reading required state from `TerminalEmulator`.
    * Implementing common rendering logic and optimizations.
    * Translating desired visual changes into calls to the **abstract drawing primitives** provided by the active `Driver` implementation.
    * **Does not contain backend-specific drawing code (e.g., Xft calls, ANSI generation).**

## 3. Component Responsibilities & Structure

*(Assuming a single crate with modules)*

* **`main.rs` (Orchestrator):**
    * Entry point, setup.
    * Instantiates `TerminalEmulator`, `AnsiProcessor`, `Driver` impl, and the **single, generic `Renderer`**.
    * Runs `epoll` loop (PTY FD, Driver FD).
    * Reads PTY -> `AnsiProcessor` -> `AnsiCommand` -> `EmulatorInput::Ansi` -> `term.interpret_event()` -> Handles `EmulatorAction`.
    * Polls Driver -> `driver.process_events()` -> `BackendEvent` -> Handles Resize/Close directly, or sends `EmulatorInput::User` -> `term.interpret_event()` -> Handles `EmulatorAction`.
    * Calls `renderer.draw()` when needed.
* **`term` Module (`TerminalEmulator` struct, `screen.rs` helpers):**
    * Defines `TerminalEmulator`, `EmulatorInput`, `EmulatorAction`.
    * Holds terminal state, including `dirty` flags. **No scrollback.**
    * `interpret_event(EmulatorInput) -> Option<EmulatorAction>`: Updates state (including setting dirty flags), signals actions. **No I/O.**
    * `resize(cols, rows)`: Updates state, marks all lines dirty.
    * State accessors (`get_dimensions`, `get_glyph`, etc.).
    * `take_dirty_lines(&mut self) -> Vec<usize>`: Returns indices of dirty lines and clears the flags.
* **`ansi` Module (`AnsiProcessor`, `lexer`, `parser`, `commands`):**
    * Parser pipeline components. Defines `AnsiCommand`.
* **`backends` Module (`Driver` trait, `BackendEvent` enum, `x11.rs`, `console.rs`):**
    * Defines `trait Driver` with **abstract drawing primitives** (e.g., `draw_text_run`, `fill_rect`, `present`, `get_font_dimensions`) and platform interaction methods (event handling, window setup, FD).
    * Defines `enum BackendEvent`.
    * `x11::XDriver`: Implements `Driver` using X11/Xft. (Formerly `XBackend`)
    * `console::ConsoleDriver`: Implements `Driver` using ANSI escape codes. (Formerly `ConsoleBackend`)
* **`renderer` Module (`Renderer` struct):**
    * **Backend-agnostic.**
    * `draw(&self, term: &mut TerminalEmulator, driver: &mut dyn Driver)`:
        * Calls `let dirty_lines = term.take_dirty_lines();`
        * If `dirty_lines` is not empty (or if a full redraw is forced):
            * Reads necessary state from `term` for dirty lines/cells.
            * Calculates required drawing operations.
            * Calls `driver.draw_text_run()`, `driver.fill_rect()`, etc.
        * Calls `driver.present()`.
    * `resize(...)`: Handles renderer-specific state updates on resize, if any.

## 4. Data & Control Flow

1.  **PTY Output:** `main` reads bytes -> `AnsiProcessor` -> `Vec<AnsiCommand>` -> `main` calls `term.interpret_event(EmulatorInput::Ansi(...))` -> `Term` state updated (incl. dirty flags) -> `main` marks for render.
2.  **User Input (Key):** `Driver` detects key -> `main` calls `driver.process_events()` -> `BackendEvent::Key` -> `main` calls `term.interpret_event(EmulatorInput::User(...))` -> `EmulatorAction::WritePty(bytes)` returned -> `main` writes bytes to PTY.
3.  **User Input (Resize):** `Driver` detects resize -> `main` calls `driver.process_events()` -> `BackendEvent::Resize` -> `main` calls `term.resize` (marks all dirty), `renderer.resize` (if needed), `ioctl`. `main` marks for render.
4.  **Rendering:** `main` detects render needed -> `main` calls `renderer.draw(&mut term, &mut *driver)` -> `Renderer` calls `term.take_dirty_lines()` -> `Renderer` calls appropriate `driver` drawing primitives based on dirty state -> `Renderer` calls `driver.present()`.

## 5. Benefits of this Architecture

* **Testability:** `TerminalEmulator` is fully testable without I/O or graphics. `Renderer` logic can be tested with mock `Driver` implementations.
* **Separation of Concerns:** Clear boundaries between state emulation, platform interaction/drawing implementation, and rendering logic.
* **Maintainability & Extensibility:** Easier to add new backends (implement `Driver`) or change rendering strategies (modify `Renderer` without touching drivers).
* **Clarity:** Interpreter/VM model is clear. `Renderer`'s backend-agnostic role and the `Driver`'s implementation role are well-defined.
* **Stability:** Avoids complex GPU rendering dependencies.

## 6. Future Considerations

* **Configuration:** Dedicated mechanism needed.
* **Error Handling:** Refine error propagation.
* **Performance:** Profile rendering calls to `Driver` primitives.
* **Platform Support:** Add more `Driver` implementations.
* **Abstract Drawing Primitives:** Carefully define the `Driver` trait's drawing methods to be expressive enough but still abstract.

