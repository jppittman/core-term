pub mod display;
pub mod input;
pub mod platform;
pub mod traits;

pub use display::messages::DriverConfig as WindowConfig;
pub use platform::EnginePlatform;
pub use traits::{AppAction, AppState, Application, EngineEvent};

/// Entry point for the Pixelflow Engine.
///
/// This function initializes the platform (windowing, display) and starts the event loop.
/// It takes an implementation of the `Application` trait, which defines the logic.
///
/// # Arguments
/// * `app` - The application logic.
/// * `config` - Window configuration.
pub fn run(app: impl Application + 'static, config: WindowConfig) -> anyhow::Result<()> {
    let platform = EnginePlatform::new(config)?;
    platform.run(app)
}
