pub mod aa;
pub mod color;
pub mod frame;
pub mod pixel;
pub mod rasterizer;

pub use aa::aa_coverage;
pub use color::{AttrFlags, Bgra8, CocoaPixel, Color, NamedColor, Rgba8, WebPixel, X11Pixel};
pub use frame::Frame;
pub use pixel::Pixel;
pub use rasterizer::{execute, execute_stripe, Rasterize, Stripe, TensorShape};
