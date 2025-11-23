// src/display/manager.rs
//! DisplayManager - Synchronous wrapper around DisplayDriver.
//!
//! Provides a simple interface for MacosPlatform to interact with the display system.
//! The manager forwards requests to the driver and stores window metrics.

use crate::display::{DisplayDriver, DriverRequest, DriverResponse};
use anyhow::{Context, Result};
use log::info;

/// Display metrics discovered during initialization.
#[derive(Debug, Clone)]
pub struct DisplayMetrics {
    pub width_px: u32,
    pub height_px: u32,
    pub scale_factor: f64,
}

/// DisplayManager manages the display driver and tracks window state.
///
/// This is a synchronous component that runs on the main thread (macOS requirement).
/// It wraps the message-based DisplayDriver API with a simple method-based interface.
pub struct DisplayManager {
    driver: Box<dyn DisplayDriver>,
    metrics: DisplayMetrics,
}

impl DisplayManager {
    /// Create a new DisplayManager with platform-specific driver.
    ///
    /// On macOS, this creates a CocoaDisplayDriver.
    pub fn new() -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            use crate::display::CocoaDisplayDriver;

            info!("DisplayManager: Creating CocoaDisplayDriver...");
            let mut driver = Box::new(CocoaDisplayDriver::new()?) as Box<dyn DisplayDriver>;

            // Initialize driver and get metrics
            info!("DisplayManager: Initializing driver...");
            let response = driver
                .handle_request(DriverRequest::Init)
                .context("Failed to initialize display driver")?;

            let metrics = match response {
                DriverResponse::InitComplete {
                    width_px,
                    height_px,
                    scale_factor,
                } => {
                    info!(
                        "DisplayManager: Initialized - {}x{} px, scale={}",
                        width_px, height_px, scale_factor
                    );
                    DisplayMetrics {
                        width_px,
                        height_px,
                        scale_factor,
                    }
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Expected InitComplete response, got {:?}",
                        response
                    ));
                }
            };

            Ok(Self { driver, metrics })
        }

        #[cfg(not(target_os = "macos"))]
        {
            Err(anyhow::anyhow!("DisplayManager: Unsupported platform"))
        }
    }

    /// Forward a request to the driver.
    ///
    /// This is the primary interface for all display operations.
    pub fn handle_request(&mut self, request: DriverRequest) -> Result<DriverResponse> {
        self.driver.handle_request(request)
    }

    /// Get current display metrics.
    pub fn metrics(&self) -> &DisplayMetrics {
        &self.metrics
    }

    /// Get window width in physical pixels.
    pub fn width_px(&self) -> u32 {
        self.metrics.width_px
    }

    /// Get window height in physical pixels.
    pub fn height_px(&self) -> u32 {
        self.metrics.height_px
    }

    /// Get display scale factor (Retina = 2.0, standard = 1.0).
    pub fn scale_factor(&self) -> f64 {
        self.metrics.scale_factor
    }
}
