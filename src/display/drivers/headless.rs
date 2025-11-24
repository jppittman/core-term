//! Headless mock display driver implementation.

use crate::display::driver::DisplayDriver;
use crate::display::messages::{
    DisplayError, DriverConfig, DriverRequest, DriverResponse,
};
use crate::platform::waker::{EventLoopWaker, NoOpWaker};
use anyhow::Result;
use log::{info, trace};

pub struct HeadlessDisplayDriver {
    width_px: u32,
    height_px: u32,
    scale_factor: f64,
    framebuffer: Option<Box<[u8]>>,
}

impl DisplayDriver for HeadlessDisplayDriver {
    fn new(config: &DriverConfig) -> Result<Self> {
        info!("HeadlessDisplayDriver::new() with config");

        let cols = config.initial_cols;
        let rows = config.initial_rows;
        let width_px = (cols * config.cell_width_px) as u32;
        let height_px = (rows * config.cell_height_px) as u32;
        let buffer_size = (width_px as usize) * (height_px as usize) * config.bytes_per_pixel;
        let framebuffer = vec![0u8; buffer_size].into_boxed_slice();

        Ok(Self {
            width_px,
            height_px,
            scale_factor: 1.0,
            framebuffer: Some(framebuffer),
        })
    }

    fn create_waker(&self) -> Box<dyn EventLoopWaker> {
        Box::new(NoOpWaker)
    }

    fn handle_request(
        &mut self,
        request: DriverRequest,
    ) -> Result<DriverResponse, DisplayError> {
        match request {
            DriverRequest::Init => {
                info!("HeadlessDisplayDriver: Init - returning metrics");
                Ok(DriverResponse::InitComplete {
                    width_px: self.width_px,
                    height_px: self.height_px,
                    scale_factor: self.scale_factor,
                })
            }
            DriverRequest::PollEvents => {
                // Mock driver returns no events
                Ok(DriverResponse::Events(Vec::new()))
            }
            DriverRequest::RequestFramebuffer => {
                let buffer = self.framebuffer.take().ok_or_else(|| {
                    anyhow::anyhow!("Framebuffer already transferred or not initialized")
                })?;
                Ok(DriverResponse::Framebuffer(buffer))
            }
            DriverRequest::Present(snapshot) => {
                trace!("HeadlessDisplayDriver: Present");
                // In a real driver, we would display the buffer.
                // Here we just accept it and return it.
                Ok(DriverResponse::PresentComplete(snapshot))
            }
            DriverRequest::SetTitle(title) => {
                info!("HeadlessDisplayDriver: SetTitle '{}'", title);
                Ok(DriverResponse::TitleSet)
            }
            DriverRequest::Bell => {
                info!("HeadlessDisplayDriver: Bell");
                Ok(DriverResponse::BellRung)
            }
            DriverRequest::SetCursorVisibility(visible) => {
                info!("HeadlessDisplayDriver: SetCursorVisibility {}", visible);
                Ok(DriverResponse::CursorVisibilitySet)
            }
            DriverRequest::CopyToClipboard(text) => {
                info!("HeadlessDisplayDriver: CopyToClipboard '{}'", text);
                Ok(DriverResponse::ClipboardCopied)
            }
            DriverRequest::RequestPaste => {
                info!("HeadlessDisplayDriver: RequestPaste");
                Ok(DriverResponse::PasteRequested)
            }
            DriverRequest::SubmitClipboardData(text) => {
                info!("HeadlessDisplayDriver: SubmitClipboardData '{}'", text);
                Ok(DriverResponse::ClipboardDataSubmitted)
            }
        }
    }
}
