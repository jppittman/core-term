//! Idiomatic pixelflow example to verify SIMD codegen
use pixelflow_core::{ManifoldExt, PARALLELISM, X, Y, materialize};

#[inline(never)]
pub fn render_circle(buffer: &mut [f32]) {
    // Circle SDF - pure algebra, no splat!
    let circle = (X * X + Y * Y).sqrt() - 100.0;

    // Render a 100x100 grid
    for y in 0..100 {
        for x_chunk in (0..100).step_by(PARALLELISM) {
            let offset = y * 100 + x_chunk;
            materialize(&circle, x_chunk as f32, y as f32, &mut buffer[offset..]);
        }
    }
}

fn main() {
    let mut buffer = vec![0.0f32; 10000];
    render_circle(&mut buffer);

    // Check center (50,50) - should be sqrt(50²+50²) - 100 = 70.7 - 100 = -29.3
    println!("Center distance: {:.1}", buffer[50 * 100 + 50]);
}
