use crate::display::messages::{DisplayControl, DisplayData, DisplayMgmt};
use actor_scheduler::Actor;
use pixelflow_core::Pixel;

/// The Platform Trait IS the Actor.
/// It manages the "HashMap" of windows and the OS connection.
pub trait Platform:
    Actor<DisplayData<Self::Pixel>, DisplayControl, DisplayMgmt> + Send + 'static
{
    /// The pixel format required by this platform.
    type Pixel: Pixel;
}
