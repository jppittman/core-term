use crate::render::color::Rgba;
use pixelflow_core::{materialize, Manifold};

/// A simple 2D image buffer.
#[derive(Clone, Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>, // RGBA data
}

impl Image {
    /// Create a new image with the given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width * height * 4],
        }
    }

    /// Render a manifold into the image.
    ///
    /// The manifold is treated as a mask (0.0 to 1.0).
    /// The image is filled with white where the manifold is 1.0,
    /// and black where it is 0.0.
    ///
    /// Future versions will support ColorManifolds.
    pub fn render_mask(&mut self, mask: &impl Manifold) {
        // Temporary buffer for single channel render
        let mut alpha = vec![0u8; self.width * self.height];

        // Render row by row
        // Note: materialize expects a buffer of size core::PARALLELISM
        // We'll process in chunks of PARALLELISM
        let mut offset = 0;
        for y in 0..self.height {
            let mut x = 0;
            while x < self.width {
                // Ensure we have enough space for a full SIMD write
                // materialize writes PARALLELISM bytes
                let chunk_size = pixelflow_core::PARALLELISM;

                // If we're at the edge, we might need a temporary buffer
                if x + chunk_size > self.width {
                    let mut temp = vec![0u8; chunk_size];
                    materialize(mask, x as f32, y as f32, &mut temp);
                    // Copy valid part
                    let valid_len = self.width - x;
                    alpha[offset + x..offset + x + valid_len].copy_from_slice(&temp[0..valid_len]);
                } else {
                    materialize(
                        mask,
                        x as f32,
                        y as f32,
                        &mut alpha[offset + x..offset + x + chunk_size],
                    );
                }
                x += chunk_size;
            }
            offset += self.width;
        }

        // Expand to RGBA (white mask)
        for (i, &a) in alpha.iter().enumerate() {
            let base = i * 4;
            self.data[base] = a; // R
            self.data[base + 1] = a; // G
            self.data[base + 2] = a; // B
            self.data[base + 3] = 255; // A (Full opaque)
        }
    }
}
