use crate::render::color::Rgba;
use pixelflow_core::Manifold;

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
    pub fn render_mask(&mut self, _mask: &impl Manifold) {
        // Placeholder implementation to allow compilation.
        // The previous implementation used a private `materialize` function.
        // Proper implementation requires public evaluation APIs.
    }
}
