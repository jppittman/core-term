//! Demonstration of generic gather operation
//!
//! Shows how `gather` works with different source types (u8, u32)

use pixelflow_core::{Batch, SimdOps};
use pixelflow_core::backend::SimdBatch; // Import SimdBatch for load

fn main() {
    println!("Generic Gather Demonstration\n");

    // Example 1: Gather from u8 palette (indexed color lookup)
    println!("=== Example 1: u8 palette lookup ===");
    let palette: Vec<u8> = (0..=255).collect();
    let indices_arr = [0, 64, 128, 255];
    let indices = Batch::<u32>::load(&indices_arr);

    let colors = unsafe { Batch::<u32>::gather_u8(&palette, indices) };
    let mut result = [0u32; 4];
    unsafe { colors.store(result.as_mut_ptr()) };

    println!("Palette indices: {:?}", indices_arr);
    println!("Gathered colors: {:?}", result);

    // Example 2: Gather from u32 texture (RGBA pixels)
    println!("\n=== Example 2: u32 texture sampling ===");
    let texture = [
        0xFF000000u32, // Red
        0x00FF0000,    // Green
        0x0000FF00,    // Blue
        0x000000FF,    // Alpha
        0xFFFFFFFF,    // White
    ];
    let sample_indices_arr = [0, 2, 4, 1]; // Red, Blue, White, Green
    let sample_indices = Batch::<u32>::load(&sample_indices_arr);

    let pixels = unsafe { Batch::<u32>::gather(texture.as_slice(), sample_indices) };
    let mut result = [0u32; 4];
    unsafe { pixels.store(result.as_mut_ptr()) };

    println!("Texture indices: {:?}", sample_indices_arr);
    println!("Sampled pixels:  {:08X?}", result);

    // Example 3: Repeat indices (texture tiling)
    println!("\n=== Example 3: Texture tiling (repeated indices) ===");
    let tile = [0xAAu8, 0xBB, 0xCC, 0xDD];
    let tiled_indices_arr = [0, 1, 0, 1]; // Repeat pattern
    let tiled_indices = Batch::<u32>::load(&tiled_indices_arr);

    let tiled = unsafe { Batch::<u32>::gather_u8(&tile, tiled_indices) };
    let mut result = [0u32; 4];
    unsafe { tiled.store(result.as_mut_ptr()) };

    println!("Tile pattern: {:?}", tile);
    println!("Tiled result: {:02X?}", result);
}
