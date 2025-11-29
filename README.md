# core-term (PixelFlow v11.0)

**`core-term` is a correct, high-performance terminal emulator powered by the PixelFlow Zero-Copy Functional Kernel.**

Its vision is to provide a robust and simple core terminal experience, utilizing a novel **Zero-Copy Functional Kernel** architecture (`pixelflow`) to achieve zero allocation per frame and static compilation of the entire render pipeline.

## Project Structure

The project is organized as a Cargo workspace with the following members:

*   **`core-term`**: The main terminal emulator application. Contains the state machine, platform layers, IO handling, and actor orchestration.
*   **`pixelflow-core`**: A high-performance, zero-cost SIMD abstraction library for pixel operations. It provides the mathematical foundation for rendering.
*   **`pixelflow-engine`**: The execution core and runtime environment.
*   **`pixelflow-render`**: The software rendering primitives (Surfaces) built on top of `pixelflow-core`.
*   **`xtask`**: Automation scripts for building and bundling the application (especially for macOS).

## Core Philosophy

`core-term` embraces several key philosophies:

*   **`st`-Inspiration:** The feature set focuses on essential terminal functionalities.
*   **Simplicity:** Complexity is managed through modular architecture rather than monolithic structures.
*   **Correctness:** Aims for accurate VT100/VT220/XTerm emulation.
*   **Zero-Copy Functional Rendering:** Prioritizes stability and predictability by using a high-performance CPU-based functional kernel (`pixelflow`), avoiding complex GPU driver dependencies while maintaining high frame rates.

## Architecture

This project implements the **PixelFlow v11.0** architecture, a synthesis of Functional Programming (Surface) and Actor Concurrency (Recycle Loop).

See [PixelFlow Architecture v11.0](docs/NORTH_STAR.md) for the complete blueprint.

Key Architectural Pillars:
* **The Monolith (Surface)**: Everything is a function `F(u, v) -> Color`.
* **Zero-Copy Recycle**: Ping-Pong buffer strategy for zero allocation per frame.
* **Engine as Compiler**: Monomorphization of the scene graph for AVX-512 optimization.

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
- **Architecture:** Please refer to [docs/NORTH_STAR.md](docs/NORTH_STAR.md) for architectural guidelines.

## License

[MIT License](LICENSE.md)
