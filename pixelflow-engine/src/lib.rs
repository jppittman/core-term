pub mod channel;
pub mod config;
pub mod display;
pub mod frame;
pub mod input;
pub mod platform;
pub mod traits;

pub use channel::{
    create_engine_channels, DriverCommand, EngineChannels, EngineCommand, EngineSender,
};
pub use config::{EngineConfig, PerformanceConfig, WindowConfig};
pub use frame::{create_frame_channel, create_recycle_channel, EngineHandle, FramePacket};
pub use platform::EnginePlatform;
pub use traits::{AppAction, AppState, Application, EngineEvent};

#[cfg(use_web_display)]
use wasm_bindgen::prelude::*;

// This code is dogshit and should be in the platform itself....
#[cfg(use_web_display)]
#[wasm_bindgen]
pub fn pixelflow_init_worker(canvas: web_sys::OffscreenCanvas, sab: js_sys::SharedArrayBuffer) {
    crate::display::drivers::web::init_resources(canvas, sab);
}

// This code is dogshit and should be in the platform itself....
#[cfg(use_web_display)]
#[wasm_bindgen]
pub fn pixelflow_dispatch_event(
    sab: js_sys::SharedArrayBuffer,
    event_val: wasm_bindgen::JsValue,
) -> Result<(), wasm_bindgen::JsValue> {
    use crate::display::drivers::web::ipc::SharedRingBuffer;
    use crate::display::DisplayEvent;

    let event: DisplayEvent = serde_wasm_bindgen::from_value(event_val).map_err(|e| {
        wasm_bindgen::JsValue::from_str(&format!("Failed to deserialize event: {}", e))
    })?;

    let ipc = SharedRingBuffer::new(&sab);
    ipc.write(&event)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&format!("Failed to write event: {}", e)))?;
    Ok(())
}

use pixelflow_core::pipe::Surface;
use pixelflow_core::Pixel;

/// Entry point for the Pixelflow Engine.
///
/// This function initializes the platform (windowing, display) and starts the event loop.
/// It takes an implementation of the `Application` trait, which defines the logic.
///
/// # Type Parameters
/// * `P` - The pixel format (e.g., `Rgba` for Cocoa, `Bgra` for X11).
///
/// # Arguments
/// * `app` - The application logic.
/// * `config` - Engine configuration.
pub fn run<P: Pixel + Surface<P>>(app: impl Application<P> + Send + 'static, config: EngineConfig) -> anyhow::Result<()> {
    let platform = EnginePlatform::new(config.into())?;
    platform.run(app)
}
