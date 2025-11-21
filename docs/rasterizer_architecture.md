# Rasterizer Compiler Architecture

## Overview

We've implemented a **zero-copy, compiler-based architecture** for terminal rendering, following the RISC philosophy where the driver is minimal and all complexity is in the compiler.

## The Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                      Renderer                            │
│  (Unchanged - generates high-level commands)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Vec<RenderCommand>                          │
│  High-level IR (like bytecode in a compiler)            │
│  - ClearAll { bg }                                       │
│  - DrawTextRun { x, y, text, fg, bg, flags, ... }        │
│  - FillRect { x, y, width, height, color, ... }          │
│  - SetWindowTitle { title }                              │
│  - RingBell                                              │
│  - PresentFrame                                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          Driver::get_framebuffer_mut()                   │
│  Driver provides mutable access to its pixel buffer      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│      rasterizer::compile_into_buffer()                   │
│  COMPILER - writes pixels directly to framebuffer        │
│  - Rasterizes text to pixels (zero-copy!)               │
│  - Converts colors to RGBA                               │
│  - Caches glyphs for performance                         │
│  - Can optimize (merge operations, skip redundant work)  │
│  - Returns only metadata commands (no pixel data!)       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Vec<DriverCommand>                          │
│  Minimal metadata commands (RISC - only 3 opcodes!)     │
│  - Present           (show the framebuffer)              │
│  - SetTitle { title }                                    │
│  - Bell                                                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                      Driver                              │
│  (Minimal - just executes metadata commands, ~100 lines)│
│  - CocoaDriver: owns framebuffer, presents to NSWindow   │
│  - X11Driver: can still use RenderCommand directly       │
│  - Future: Wayland, headless, GPU drivers                │
└─────────────────────────────────────────────────────────┘
```

## Benefits

1. **Zero-copy rendering** - Rasterizer writes directly to driver's framebuffer
2. **Tiny messages** - DriverCommands contain no pixel data (just metadata)
3. **Trivial drivers** - Drivers only need ~100 lines (vs ~1000+ for full drivers)
4. **Easy to optimize** - All optimizations in one place (the rasterizer)
5. **Easy to test** - Pure function that writes to a buffer
6. **Async-ready** - Small metadata commands are cheap to send across threads
7. **Universal** - All platforms benefit from rasterizer optimizations

## Architecture Details

### Driver Trait - Framebuffer API

```rust
pub trait Driver {
    /// Returns mutable access to the driver's framebuffer
    /// Format: Linear RGBA array, row-major layout
    fn get_framebuffer_mut(&mut self) -> &mut [u8];

    /// Returns framebuffer dimensions in pixels
    fn get_framebuffer_size(&self) -> (usize, usize);

    // ... other methods ...
}
```

### DriverCommand - Minimal Metadata

```rust
pub enum DriverCommand {
    /// Present the framebuffer to the screen
    Present,
    /// Set window title
    SetTitle { title: String },
    /// Ring bell
    Bell,
}
```

### Rasterizer - The Compiler

```rust
/// Compiles high-level RenderCommands into pixels, writing directly
/// to the provided framebuffer. Returns only metadata commands.
pub fn compile_into_buffer(
    commands: Vec<RenderCommand>,
    framebuffer: &mut [u8],           // Direct write target
    buffer_width_px: usize,
    buffer_height_px: usize,
    cell_width_px: usize,
    cell_height_px: usize,
) -> Vec<DriverCommand>  // Only metadata, no pixel data!
```

## Current Status

### ✅ Complete

1. **DriverCommand enum** - 3 minimal metadata commands
2. **Driver trait framebuffer API**:
   - `get_framebuffer_mut()` - Returns mutable framebuffer access
   - `get_framebuffer_size()` - Returns dimensions
3. **Rasterizer/Compiler**:
   - `rasterizer::compile_into_buffer()` function implemented
   - Zero-copy: writes directly to driver's framebuffer
   - Software glyph rendering (placeholder, can be replaced with real font engine)
   - Color conversion (Named/RGB/Indexed → RGBA)
   - Glyph caching infrastructure
4. **Unit Tests** - 15 comprehensive tests covering:
   - All command types
   - Edge cases
   - Different cell sizes
   - Color conversions
   - Framebuffer writes
   - Tests follow project guidelines (public API only)
5. **All Drivers Updated**:
   - **CocoaDriver**: Owns real framebuffer, ready for pixel blitting
   - **X11Driver**: Dummy framebuffer (still uses Xft for now)
   - **ConsoleDriver**: Dummy framebuffer (uses ANSI codes)
   - **WaylandDriver**: Dummy framebuffer (stub implementation)
   - **MockDriver**: Real framebuffer for testing

### 🚧 TODO

1. **CocoaDriver pixel display**:
   - Convert framebuffer to CGImage
   - Display via NSView's drawRect or CALayer
   - Handle retina/HiDPI displays

2. **CocoaDriver event handling**:
   - Implement non-blocking event polling
   - Convert NSEvent to BackendEvent
   - Handle keyboard input
   - Handle mouse events
   - Handle window resize

3. **Wire up rasterizer in MacosPlatform**:
   ```rust
   let render_commands = renderer.prepare_render_commands(...);
   let framebuffer = driver.get_framebuffer_mut();
   let (width, height) = driver.get_framebuffer_size();
   let driver_commands = rasterizer::compile_into_buffer(
       render_commands,
       framebuffer,
       width,
       height,
       cell_width,
       cell_height
   );
   driver.execute_driver_commands(driver_commands)?;
   ```

4. **Testing**:
   - Fix X11 conditional compilation for macOS
   - Test CocoaDriver with actual window
   - Test end-to-end: Renderer → Rasterizer → CocoaDriver

## Notes

- The `cocoa` crate is deprecated but still works (warnings only)
- Eventually migrate to `objc2-app-kit` for long-term support
- Software rasterizer is currently a placeholder (filled rectangles)
- Can be replaced with real font rendering (fontdue, ab_glyph, etc.)
- X11Driver can be migrated to use framebuffer API in the future

## Architecture Analogy

This is like a traditional compiler:

- **Renderer** = Frontend (parser, generates IR)
- **RenderCommand** = Intermediate Representation (SSA, bytecode)
- **rasterizer::compile_into_buffer()** = Middle-end + Backend (optimization, code gen)
- **DriverCommand** = Assembly/Machine Code (metadata only!)
- **Driver::framebuffer** = Memory (where the "machine code" writes)
- **Driver** = CPU (executes metadata instructions)

Just as a compiler can target multiple architectures (x86, ARM, RISC-V),
our compiler can target multiple drivers (Cocoa, X11, Wayland, headless).

## Performance Characteristics

- **Zero copies**: Pixels written once, directly to final destination
- **Small messages**: DriverCommands are ~24 bytes each (vs 100s of KB with pixel data)
- **Cache-friendly**: Glyph cache in rasterizer benefits all platforms
- **Async-ready**: Metadata commands are cheap to send across threads/channels
- **Optimal**: All rendering optimizations benefit all platforms
