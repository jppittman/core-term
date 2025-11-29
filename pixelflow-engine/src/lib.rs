pub mod display;
pub mod frame;
pub mod input;
pub mod platform;
pub mod traits;

pub use display::messages::DriverConfig as WindowConfig;
pub use frame::{create_frame_channel, create_recycle_channel, EngineHandle, FramePacket};
pub use platform::EnginePlatform;
pub use traits::{AppAction, AppState, Application, EngineEvent};

#[cfg(use_web_display)]
use wasm_bindgen::prelude::*;

#[cfg(use_web_display)]
#[wasm_bindgen]
pub fn pixelflow_init_worker(canvas: web_sys::OffscreenCanvas, sab: js_sys::SharedArrayBuffer) {
    crate::display::drivers::web::init_resources(canvas, sab);
}

#[cfg(use_web_display)]
#[wasm_bindgen]
pub fn pixelflow_dispatch_event(
    sab: js_sys::SharedArrayBuffer,
    event_val: wasm_bindgen::JsValue,
) -> Result<(), wasm_bindgen::JsValue> {
    use crate::display::drivers::web::ipc::SharedRingBuffer;
    use crate::display::DisplayEvent;

    let event: DisplayEvent = serde_wasm_bindgen::from_value(event_val)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&format!("Failed to deserialize event: {}", e)))?;

    let ipc = SharedRingBuffer::new(&sab);
    ipc.write(&event)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&format!("Failed to write event: {}", e)))?;
    Ok(())
}

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
