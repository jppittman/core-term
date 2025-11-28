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
    /// Create a new DisplayManager with display driver selected at build time.
    pub fn new(driver_config: crate::display::messages::DriverConfig) -> Result<Self> {
        // Create driver based on build-time selection
        #[cfg(use_cocoa_display)]
        {
            use crate::display::drivers::CocoaDisplayDriver;
            info!("DisplayManager: Creating CocoaDisplayDriver with config...");
            let mut driver =
                Box::new(CocoaDisplayDriver::new(&driver_config)?) as Box<dyn DisplayDriver>;

            info!("DisplayManager: Calling Init...");
            let response = driver
                .handle_request(DriverRequest::Init)
                .map_err(|e| anyhow::anyhow!(e))
                .context("Failed to initialize display driver")?;

            let metrics = Self::extract_metrics(response)?;
            Ok(Self { driver, metrics })
        }

        #[cfg(use_x11_display)]
        {
            use crate::display::drivers::X11DisplayDriver;
            info!("DisplayManager: Creating X11DisplayDriver with config...");
            let mut driver =
                Box::new(X11DisplayDriver::new(&driver_config)?) as Box<dyn DisplayDriver>;

            info!("DisplayManager: Calling Init...");
            let response = driver
                .handle_request(DriverRequest::Init)
                .map_err(|e| anyhow::anyhow!(e))
                .context("Failed to initialize display driver")?;

            let metrics = Self::extract_metrics(response)?;
            Ok(Self { driver, metrics })
        }

        #[cfg(use_headless_display)]
        {
            use crate::display::drivers::HeadlessDisplayDriver;
            info!("DisplayManager: Creating HeadlessDisplayDriver with config...");
            let mut driver =
                Box::new(HeadlessDisplayDriver::new(&driver_config)?) as Box<dyn DisplayDriver>;

            info!("DisplayManager: Calling Init...");
            let response = driver
                .handle_request(DriverRequest::Init)
                .map_err(|e| anyhow::anyhow!(e))
                .context("Failed to initialize display driver")?;

            let metrics = Self::extract_metrics(response)?;
            Ok(Self { driver, metrics })
        }
    }

    /// Helper to extract metrics from InitComplete response
    fn extract_metrics(response: DriverResponse) -> Result<DisplayMetrics> {
        match response {
            DriverResponse::InitComplete {
                width_px,
                height_px,
                scale_factor,
            } => {
                info!(
                    "DisplayManager: Initialized - {}x{} px, scale={}",
                    width_px, height_px, scale_factor
                );
                Ok(DisplayMetrics {
                    width_px,
                    height_px,
                    scale_factor,
                })
            }
            _ => Err(anyhow::anyhow!(
                "Expected InitComplete response, got {:?}",
                response
            )),
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

    pub fn create_waker(&self) -> Box<dyn crate::platform::waker::EventLoopWaker> {
        self.driver.create_waker()
    }
}
