// src/display/manager.rs
//! DisplayManager - Synchronous wrapper around DisplayDriver.

// FIX: Import DisplayError
use crate::display::driver::DisplayDriver;
use crate::display::messages::{DisplayError, DriverRequest, DriverResponse};
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
pub struct DisplayManager {
    driver: Box<dyn DisplayDriver>,
    metrics: DisplayMetrics,
}

impl DisplayManager {
    /// Create a new DisplayManager with platform-specific driver.
    pub fn new() -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            use crate::config::CONFIG;
            use crate::display::messages::DriverConfig;
            use crate::display::CocoaDisplayDriver;

            info!("DisplayManager: Creating CocoaDisplayDriver...");
            let mut driver = Box::new(CocoaDisplayDriver::new()?) as Box<dyn DisplayDriver>;

            // Build DriverConfig from CONFIG
            let driver_config = DriverConfig {
                initial_window_x: 100.0,
                initial_window_y: 100.0,
                initial_cols: CONFIG.appearance.columns as usize,
                initial_rows: CONFIG.appearance.rows as usize,
                cell_width_px: CONFIG.appearance.cell_width_px,
                cell_height_px: CONFIG.appearance.cell_height_px,
                bytes_per_pixel: 4,
                bits_per_component: 8,
                bits_per_pixel: 32,
                max_draw_latency_seconds: CONFIG.performance.max_draw_latency_ms.as_secs_f64(),
            };

            info!("DisplayManager: Initializing driver...");
            // FIX: Convert DisplayError to anyhow::Error using map_err for Init
            let response = driver
                .handle_request(DriverRequest::Init(driver_config))
                .map_err(|e| anyhow::anyhow!(e))
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
            use crate::config::CONFIG;
            use crate::display::messages::DriverConfig;
            use crate::display::drivers::HeadlessDisplayDriver;

            info!("DisplayManager: Creating HeadlessDisplayDriver...");
            let mut driver = Box::new(HeadlessDisplayDriver::new()?) as Box<dyn DisplayDriver>;

            // Build DriverConfig from CONFIG
            let driver_config = DriverConfig {
                initial_window_x: 100.0,
                initial_window_y: 100.0,
                initial_cols: CONFIG.appearance.columns as usize,
                initial_rows: CONFIG.appearance.rows as usize,
                cell_width_px: CONFIG.appearance.cell_width_px,
                cell_height_px: CONFIG.appearance.cell_height_px,
                bytes_per_pixel: 4,
                bits_per_component: 8,
                bits_per_pixel: 32,
                max_draw_latency_seconds: CONFIG.performance.max_draw_latency_ms.as_secs_f64(),
            };

            info!("DisplayManager: Initializing driver...");
            let response = driver
                .handle_request(DriverRequest::Init(driver_config))
                .map_err(|e| anyhow::anyhow!(e))
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
    }

    /// Forward a request to the driver.
    pub fn handle_request(
        &mut self,
        request: DriverRequest,
    ) -> Result<DriverResponse, DisplayError> {
        // FIX: The driver returns Result<DriverResponse, DisplayError>. Convert to anyhow::Result.
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
