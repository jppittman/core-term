// Public API - organized under api module
pub mod api;

// Internal modules
pub mod channel;
pub mod config;
pub mod display;
pub mod frame;
pub mod input;
pub mod platform;
pub mod render_pool;
pub mod traits;
pub mod vsync_actor;

// Re-export public API types at crate root (new, preferred)
pub use api::public::*;

// Re-export priority-channel as actor module
/// Actor model primitives (message passing, scheduling, priority lanes)
pub use actor_scheduler as actor;

// Convenience re-exports at crate root (for backward compatibility)
pub use actor_scheduler::{
    ActorHandle, ActorScheduler, Message, SchedulerHandler, SendError, WakeHandler,
};

// Make private API available throughout crate (not exported)
#[allow(unused_imports)]
use api::private::*;

// Re-export legacy types for backward compatibility (deprecated)
pub use channel::{
    create_engine_channels, DriverCommand, EngineActorHandle, EngineActorScheduler,
    EngineChannels, EngineControl, AppManagement, EngineSender,
};
pub use config::{EngineConfig, PerformanceConfig, WindowConfig};
pub use frame::{create_frame_channel, create_recycle_channel, EngineHandle, FramePacket};
pub use platform::EnginePlatform;

#[cfg(use_web_display)]
use wasm_bindgen::prelude::*;

// This code is dogshit and should be in the platform itself....
#[cfg(use_web_display)]
#[wasm_bindgen]
pub fn pixelflow_init_worker(canvas: web_sys::OffscreenCanvas, sab: js_sys::SharedArrayBuffer, scale_factor: f64) {
    crate::display::drivers::web::init_resources(canvas, sab, scale_factor);
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

pub use platform::PlatformPixel;

/// Entry point for the Pixelflow Engine.
///
/// This function initializes the platform (windowing, display) and starts the event loop.
/// It takes an implementation of the `Application` trait, which defines the logic,
/// and an engine actor handle for the app to send responses back.
///
/// The pixel type is determined by the platform:
/// - Cocoa (macOS): `Rgba`
/// - X11 (Linux): `Bgra`
/// - Web: `Rgba`
///
/// # Arguments
/// * `app` - The application logic implementing `Application`.
/// * `engine_handle` - Handle for app to send responses back to engine (renders, actions).
/// * `config` - Engine configuration.
pub fn run(
    app: impl Application + Send + 'static,
    engine_handle: api::private::EngineActorHandle<PlatformPixel>,
    config: EngineConfig,
) -> anyhow::Result<()> {
    let platform = EnginePlatform::new(app, engine_handle, config)?;
    platform.run()
}
