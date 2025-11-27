# core-term

**`core-term` is a correct, reasonably performant, and maintainable terminal emulator written in Rust.**

Its vision is to provide a robust and simple core terminal experience, drawing inspiration from the `st` (simple terminal) project while being designed from the ground up for **extendibility** and **modern architecture**.

## Project Structure

The project is organized as a Cargo workspace with the following members:

*   **`core-term`**: The main terminal emulator application. Contains the state machine, platform layers, IO handling, and actor orchestration.
*   **`pixelflow-core`**: A high-performance, zero-cost SIMD abstraction library for pixel operations. It provides the mathematical foundation for rendering.
*   **`pixelflow-render`**: The software rendering engine built on top of `pixelflow-core`. Handles software rasterization, glyph rendering, and blitting.
*   **`xtask`**: Automation scripts for building and bundling the application (especially for macOS).

## Core Philosophy

`core-term` embraces several key philosophies:

*   **`st`-Inspiration:** The feature set focuses on essential terminal functionalities.
*   **Simplicity:** Complexity is managed through modular architecture (Actor Model) rather than monolithic structures.
*   **Correctness:** Aims for accurate VT100/VT220/XTerm emulation.
*   **Software Rendering:** Prioritizes stability and predictability by using a high-performance CPU-based renderer (`pixelflow`), avoiding complex GPU driver dependencies while maintaining high frame rates.

## Architecture

`core-term` uses an **Actor Model** architecture to ensure concurrency safety and clean separation of concerns:

*   **`Orchestrator`**: The central hub. It routes messages between the PTY, the Display/Renderer, and the Terminal Emulator.
*   **`TerminalEmulator`**: A pure state machine. It accepts inputs (ANSI bytes, User actions) and produces side-effects (Draw, Write to PTY). It does not perform I/O directly.
*   **`EventMonitorActor`**: Manages PTY I/O. It spawns dedicated read/write threads to ensure non-blocking operations using `kqueue` (macOS) or `epoll` (Linux).
*   **`Renderer`**: Runs on a dedicated thread. It receives snapshots of the terminal state and produces frames using `pixelflow-render`.
*   **`Platform`**: Handles window creation, input events, and displaying the rendered framebuffer.

## Building and Running

### Prerequisites

You will need Rust (stable) installed.

On Linux, you need X11 development headers:
```bash
sudo apt-get install libx11-dev libxext-dev libxft-dev libfontconfig1-dev libfreetype6-dev libxkbcommon-dev
```

### Building

Standard cargo build:
```bash
cargo build --release
```

### Running

To run the terminal:
```bash
cargo run --release
```

### Bundling (macOS)

We provide an `xtask` to bundle the application into a `.app` for macOS:

```bash
cargo xtask bundle-run
```
This will build `CoreTerm.app`, place it in the root directory, and launch it.

## Configuration

`core-term` looks for configuration or uses sensible defaults.
(Currently, configuration is mostly code-defined or loaded from defaults, see `core-term/src/config.rs`).

## Contributing

Contributions are welcome!
- **Code Style:** Follow Rust standard style. Docstrings are required for all public items.
- **Architecture:** Respect the actor model separation. Logic should reside in the appropriate actor/module.

## License

[MIT License](LICENSE.md)
