//! Demonstration of generic gather operation
//!
//! Shows how `gather` works with different source types (u8, u16, u32)

use pixelflow_core::Batch;

fn main() {
    println!("Generic Gather Demonstration\n");

    // Example 1: Gather from u8 palette (indexed color lookup)
    println!("=== Example 1: u8 palette lookup ===");
    let palette: Vec<u8> = (0..=255).collect();
    let indices = [0, 64, 128, 255];

    let colors = unsafe { Batch::<u32>::gather(&palette, indices) };
    let mut result = [0u32; 4];
    unsafe { colors.store(result.as_mut_ptr()) };

    println!("Palette indices: {:?}", indices);
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
    let sample_indices = [0, 2, 4, 1]; // Red, Blue, White, Green

    let pixels = unsafe { Batch::<u32>::gather(texture.as_slice(), sample_indices) };
    let mut result = [0u32; 4];
    unsafe { pixels.store(result.as_mut_ptr()) };

    println!("Texture indices: {:?}", sample_indices);
    println!("Sampled pixels:  {:08X?}", result);

    // Example 3: Gather from u16 (e.g., depth buffer or compressed texture)
    println!("\n=== Example 3: u16 depth buffer ===");
    let depth_buffer = [0u16, 1000, 2000, 3000, 4000, 5000];
    let depth_indices = [1, 3, 5, 2];

    let depths = unsafe { Batch::<u32>::gather(&depth_buffer, depth_indices) };
    let mut result = [0u32; 4];
    unsafe { depths.store(result.as_mut_ptr()) };

    println!("Depth indices: {:?}", depth_indices);
    println!("Depth values:  {:?}", result);

    // Example 4: Repeat indices (texture tiling)
    println!("\n=== Example 4: Texture tiling (repeated indices) ===");
    let tile = [0xAAu8, 0xBB, 0xCC, 0xDD];
    let tiled_indices = [0, 1, 0, 1]; // Repeat pattern

    let tiled = unsafe { Batch::<u32>::gather(&tile, tiled_indices) };
    let mut result = [0u32; 4];
    unsafe { tiled.store(result.as_mut_ptr()) };

    println!("Tile pattern: {:?}", tile);
    println!("Tiled result: {:02X?}", result);

    println!("\n✓ Generic gather works with u8, u16, u32, and any Copy + Into<u32> type!");
}
