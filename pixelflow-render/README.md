# PixelFlow Render

**PixelFlow Render** is a high-performance, software-based rendering engine built on top of `pixelflow-core`. It provides a flexible API for 2D graphics, including primitive rasterization, text rendering, and frame management.

## Key Features

*   **Software Rasterization**: Efficient CPU-based rendering leveraging SIMD optimizations from `pixelflow-core`.
*   **Primitive Operations**: Supports drawing rectangles, blitting images, and clearing buffers via the `Op` enum.
*   **Text Rendering**: Integrated text rendering with support for font atlases and glyph positioning.
*   **Command-Based API**: Stateless rendering via a list of `Op` commands, allowing for batch processing and easy integration.

## Coordinate Systems

*   **Pixel Coordinates**: Used by `Op::Blit` and similar operations. These map directly to the framebuffer pixels.
*   **Grid Coordinates**: Used by `Op::Text`. The `x` and `y` parameters represent column and row indices, which are multiplied by `cell_width` and `cell_height` to determine the pixel position.

## Example Usage

Here's how to use `pixelflow-render` to process a frame:

```rust
use pixelflow_render::{process_frame, Color, NamedColor, Op};

fn main() {
    let width = 800;
    let height = 600;
    let cell_width = 10;
    let cell_height = 20;
    let mut framebuffer = vec![0u32; width * height];

    let ops: Vec<Op<&[u8]>> = vec![
        // Clear the screen to Blue
        Op::Clear {
            color: Color::Named(NamedColor::Blue),
        },
        // Draw text at grid column 1, row 1
        Op::Text {
            ch: 'A',
            x: 1, // Grid column
            y: 1, // Grid row
            fg: Color::Named(NamedColor::White),
            bg: Color::Named(NamedColor::Black),
        },
    ];

    // Process the frame
    process_frame(
        &mut framebuffer,
        width,
        height,
        cell_width,
        cell_height,
        &ops
    );
}
```

## Modules

*   **`commands`**: Defines the `Op` enum and rendering commands.
*   **`glyph`**: Handles font loading, atlas management, and glyph rendering.
*   **`rasterizer`**: Contains the core logic for processing frames and executing commands.
*   **`types`**: Common data structures like `Color`.
