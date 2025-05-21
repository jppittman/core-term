# core-term

**`core-term` is a correct, reasonably performant, and maintainable terminal emulator written in Rust.**

Its vision is to provide a robust and simple core terminal experience, drawing inspiration from the `st` (simple terminal) project while being designed from the ground up for **extendibility**. Where `st` is designed to be simple by direct modification of its source code, `core-term` aims to be an "extendible core terminal" by providing a clear architecture that allows for new backends and customization through well-defined interfaces.

## Core Philosophy

`core-term` embraces several key philosophies:

*   **`st`-Inspiration:** The initial feature set and core emulation goals are aligned with `st`, focusing on essential terminal functionalities.
*   **Suckless Principles:** We adhere to the suckless philosophy of simplicity and encouraging composition with dedicated tools. Features that can be better handled by external programs (like `tmux` for scrollback) are intentionally kept out of scope.
*   **Focus on Core Terminal Emulation:** The primary goal is to excel at emulating standard terminal behaviors (VT100/VT220/XTerm compatibility) correctly and efficiently.

## Key Features

*   **VT100/VT220/XTerm Compatibility (Target):** Aiming for comprehensive support for common terminal control sequences and behaviors.
*   **CPU-Based Rendering:** Prioritizes stability and simplicity by using well-established, CPU-based rendering techniques, avoiding complex GPU driver dependencies.
*   **Modular and Testable Design:** Built with a clear separation of concerns, making components independently testable and the overall system easier to maintain and understand.

## Extendibility: The "Core" Difference

A key differentiator for `core-term` is its emphasis on extendibility. Unlike `st`, which typically requires direct source modification for customization, `core-term` is architected to be a foundational "core" that can be readily extended.

The primary mechanism for this is the **`Driver` trait**. This interface allows developers to implement new rendering and input backends for different platforms or display systems (e.g., Wayland, a different windowing toolkit, or even a custom embedded environment) without altering the core terminal emulation logic. This makes `core-term` a versatile foundation for a variety of terminal-based applications and environments.

## Architecture Overview

`core-term` adopts an **Interpreter/Virtual Machine (VM) model** for its central logic, promoting modularity and clear responsibilities:

*   **`TerminalEmulator` (Interpreter/VM):** The heart of the terminal. It's a self-contained state machine that processes inputs (derived from PTY output and user actions), updates its internal state (grid, cursor, attributes), and signals required external side-effects. It does not perform I/O or rendering directly.
*   **`AppOrchestrator`:** The central coordinator. It manages the main event loop, PTY I/O, drives the PTY parser, feeds inputs to the `TerminalEmulator`, executes actions signaled by the emulator, and orchestrates the rendering process.
*   **`Renderer`:** A backend-agnostic component. It requests a snapshot of the terminal state from the `TerminalEmulator` and translates this into a list of abstract `RenderCommand`s, which are platform-agnostic drawing instructions.
*   **`Driver` (Trait):** An interface implemented by backend-specific components (e.g., `XDriver` for X11). It translates platform-specific input events into abstract `BackendEvent`s for the emulator and executes abstract `RenderCommand`s from the `Renderer` using platform-specific APIs.

This architecture ensures a strong separation of concerns: terminal logic is isolated in `TerminalEmulator`, rendering logic in `Renderer`, and platform interaction in `Driver` implementations.

## Features Out of Scope

In line with the project's philosophy of simplicity and encouraging composition with dedicated tools, the following features are explicitly out of scope:

*   **Built-in scrollback:** Users are encouraged to use tools like `tmux` or `screen`.
*   **Advanced font rendering:** Ligatures, complex text shaping, and other advanced font features beyond a monospaced grid are not planned.
*   **Image support:** Or other graphical extensions beyond standard terminal capabilities.

## Ethos/Design Principles

The development of `core-term` is guided by the following principles:

*   **Simplicity:** Strive for the simplest possible solution that meets requirements.
*   **Stability:** Prioritize robust and reliable operation.
*   **Maintainability:** Write clear, well-organized code that is easy to understand and modify.
*   **Clarity:** Emphasize clear code over excessive comments. Code should be self-explanatory where possible. (See `docs/STYLE.md`).
*   **Correctness:** Focus on accurate terminal emulation.

## Building and Running

`core-term` is a standard Rust project and can be built using Cargo:

```bash
cargo build
```

To run `core-term` (assuming an X11 backend is the default or only one available):

```bash
cargo run
```

**Dependencies:**
For building, especially the X11 backend, you will likely need the Xlib development headers. On Debian/Ubuntu-based systems, these can typically be installed via:
```bash
sudo apt-get install -y libx11-dev
```
Other platforms will have equivalent packages for their X11 development libraries.

## Contributing

Contributions are welcome! Before contributing, please familiarize yourself with:

*   **Design Document (`docs/NORTH_STAR.md`):** Understand the project's architecture and vision.
*   **Style Guide (`docs/STYLE.md`):** Adhere to the coding style and conventions.

Please ensure your contributions align with the project's philosophy and design principles.
