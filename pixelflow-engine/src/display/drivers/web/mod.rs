pub mod ipc;
use ipc::SharedRingBuffer;
use crate::display::driver::DisplayDriver;
use crate::display::messages::{
    DisplayError, DriverConfig, DriverRequest, DriverResponse,
};
use crate::platform::waker::{EventLoopWaker, NoOpWaker};
use anyhow::{anyhow, Result};
use std::cell::RefCell;
use web_sys::{OffscreenCanvas, OffscreenCanvasRenderingContext2d, ImageData};
use wasm_bindgen::{Clamped, JsCast};

thread_local! {
    static RESOURCES: RefCell<Option<(OffscreenCanvas, js_sys::SharedArrayBuffer)>> = RefCell::new(None);
}

pub fn init_resources(canvas: OffscreenCanvas, sab: js_sys::SharedArrayBuffer) {
    RESOURCES.with(|r| *r.borrow_mut() = Some((canvas, sab)));
}

pub struct WebDisplayDriver {
    context: OffscreenCanvasRenderingContext2d,
    ipc: SharedRingBuffer,
    width_px: u32,
    height_px: u32,
    framebuffer: Option<Box<[u8]>>,
}

impl DisplayDriver for WebDisplayDriver {
    fn new(config: &DriverConfig) -> Result<Self> {
        let (canvas, sab) = RESOURCES.with(|r| {
            r.borrow_mut()
                .take()
                .ok_or_else(|| anyhow!("Web resources not initialized. Call init_resources() first."))
        })?;

        let context = canvas
            .get_context("2d")
            .map_err(|_| anyhow!("Failed to get 2d context"))?
            .ok_or_else(|| anyhow!("Context is null"))?
            .dyn_into::<OffscreenCanvasRenderingContext2d>()
            .map_err(|_| anyhow!("Failed to cast context"))?;

        let ipc = SharedRingBuffer::new(&sab);

        let width_px = (config.initial_cols * config.cell_width_px) as u32;
        let height_px = (config.initial_rows * config.cell_height_px) as u32;

        // Resize canvas
        canvas.set_width(width_px);
        canvas.set_height(height_px);

        let buffer_size = (width_px as usize) * (height_px as usize) * 4;
        let framebuffer = vec![0u8; buffer_size].into_boxed_slice();

        Ok(Self {
            context,
            ipc,
            width_px,
            height_px,
            framebuffer: Some(framebuffer),
        })
    }

    fn create_waker(&self) -> Box<dyn EventLoopWaker> {
        Box::new(NoOpWaker)
    }

    fn handle_request(&mut self, request: DriverRequest) -> std::result::Result<DriverResponse, DisplayError> {
         match request {
            DriverRequest::Init => {
                Ok(DriverResponse::InitComplete {
                    width_px: self.width_px,
                    height_px: self.height_px,
                    scale_factor: 1.0,
                })
            }
            DriverRequest::PollEvents => {
                match self.ipc.blocking_read_timeout(16) {
                    Ok(Some(evt)) => Ok(DriverResponse::Events(vec![evt])),
                    Ok(None) => Ok(DriverResponse::Events(vec![])),
                    Err(e) => Err(DisplayError::Generic(e)),
                }
            }
            DriverRequest::RequestFramebuffer => {
                 let buffer = self.framebuffer.take().ok_or_else(|| anyhow!("No framebuffer"))?;
                 Ok(DriverResponse::Framebuffer(buffer))
            }
            DriverRequest::Present(snapshot) => {
                let data = snapshot.framebuffer.as_ref();
                let image_data = ImageData::new_with_u8_clamped_array_and_sh(
                    Clamped(data),
                    snapshot.width_px,
                    snapshot.height_px,
                ).map_err(|e| DisplayError::Generic(anyhow!("Failed to create ImageData: {:?}", e)))?;

                self.context.put_image_data(&image_data, 0.0, 0.0)
                    .map_err(|e| DisplayError::Generic(anyhow!("Failed to put image data: {:?}", e)))?;

                Ok(DriverResponse::PresentComplete(snapshot))
            }
            // Stubs
            DriverRequest::SetTitle(_) => Ok(DriverResponse::TitleSet),
            DriverRequest::Bell => Ok(DriverResponse::BellRung),
            DriverRequest::SetCursorVisibility(_) => Ok(DriverResponse::CursorVisibilitySet),
            DriverRequest::CopyToClipboard(_) => Ok(DriverResponse::ClipboardCopied),
            DriverRequest::RequestPaste => Ok(DriverResponse::PasteRequested),
            DriverRequest::SubmitClipboardData(_) => Ok(DriverResponse::ClipboardDataSubmitted),
         }
    }
}
