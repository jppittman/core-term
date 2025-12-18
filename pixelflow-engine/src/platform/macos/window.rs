use crate::api::public::WindowDescriptor;
use crate::platform::macos::cocoa::{NSPoint, NSRect, NSSize, NSView, NSWindow};
use crate::platform::macos::sys::{self, Id, BOOL, YES};
use anyhow::Result;
use std::ffi::c_void;

pub struct MacWindow {
    pub(crate) window: NSWindow,
    pub(crate) view: NSView,
    pub(crate) layer: Id, // CAMetalLayer
    pub(crate) current_width: u32,
    pub(crate) current_height: u32,
}

impl MacWindow {
    pub fn new(desc: WindowDescriptor) -> Result<Self> {
        let rect = NSRect::new(
            NSPoint::new(0.0, 0.0),
            NSSize::new(desc.width as f64, desc.height as f64),
        );

        // NSWindowStyleMask: Titled | Closable | Miniaturizable | Resizable
        let style_mask = 1 | 2 | 4 | 8;
        // NSBackingStoreBuffered = 2
        let backing = 2;

        let window = NSWindow::alloc().init_with_content_rect(rect, style_mask, backing, false);
        window.set_title(&desc.title);

        let view = NSView::alloc().init_with_frame(rect);

        // Create Metal Layer
        let layer = unsafe {
            let cls = sys::class(b"CAMetalLayer\0");
            sys::send(cls, sys::sel(b"layer\0"))
        };

        unsafe {
            // [layer setDevice: MTLCreateSystemDefaultDevice()]
            #[link(name = "Metal", kind = "framework")]
            extern "C" {
                fn MTLCreateSystemDefaultDevice() -> Id;
            }
            let device = MTLCreateSystemDefaultDevice();
            if device.is_null() {
                anyhow::bail!("Failed to create Metal device");
            }
            sys::send_1::<(), Id>(layer, sys::sel(b"setDevice:\0"), device);

            // [layer setPixelFormat: 70 (RGBA8Unorm)]
            sys::send_1::<(), u64>(layer, sys::sel(b"setPixelFormat:\0"), 70);

            // [layer setFramebufferOnly: YES] - optimization
            sys::send_1::<(), BOOL>(layer, sys::sel(b"setFramebufferOnly:\0"), YES);

            // [view setLayer: layer]
            sys::send_1::<(), Id>(view.0, sys::sel(b"setLayer:\0"), layer);

            // [view setWantsLayer: YES]
            view.set_wants_layer(true);
        }

        window.set_content_view(view);
        window.make_key_and_order_front();

        // Center window?
        unsafe {
            sys::send::<()>(window.0, sys::sel(b"center\0"));
        }

        Ok(Self {
            window,
            view,
            layer,
            current_width: desc.width,
            current_height: desc.height,
        })
    }
}

impl MacWindow {
    pub fn set_title(&mut self, title: &str) {
        self.window.set_title(title);
    }

    pub fn set_size(&mut self, width: u32, height: u32) {
        let size = sys::CGSize {
            width: width as f64,
            height: height as f64,
        };
        unsafe {
            // Need to get current origin to keep it in place, or just set size?
            // "setContentSize:" is easier for content variance.
            sys::send_1::<(), sys::CGSize>(self.window.0, sys::sel(b"setContentSize:\0"), size);
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.current_width, self.current_height)
    }

    pub fn scale_factor(&self) -> f64 {
        unsafe {
            let scale: f64 = sys::send(self.window.0, sys::sel(b"backingScaleFactor\0"));
            scale
        }
    }

    pub fn set_cursor(&mut self, _icon: crate::api::public::CursorIcon) {
        // TODO: Implement set_cursor
    }

    pub fn set_visible(&mut self, visible: bool) {
        if visible {
            self.window.make_key_and_order_front();
        } else {
            unsafe {
                sys::send::<()>(self.window.0, sys::sel(b"orderOut:\0"));
            }
        }
    }

    pub fn request_redraw(&mut self) {
        unsafe {
            sys::send_1::<(), BOOL>(self.view.0, sys::sel(b"setNeedsDisplay:\0"), YES);
        }
    }

    pub fn present(&mut self, frame: pixelflow_render::Frame<pixelflow_render::color::Rgba>) {
        // Metal presentation logic.
        unsafe {
            // Ensure drawable size matches frame
            sys::send_1::<(), sys::MTLSize>(
                self.layer,
                sys::sel(b"setDrawableSize:\0"),
                sys::MTLSize {
                    width: frame.width as usize,
                    height: frame.height as usize,
                    depth: 0, // Unused for CGSize/NSSize equivalent mapping
                },
            );
            // Note: setDrawableSize takes CGSize. MTLSize has 3 fields.
            // CGSize is {width, height}. MTLSize is {width, height, depth}.
            // DIRECTLY sending MTLSize might be wrong if ABI expects CGSize.
            // On ARM64:
            // CGSize (2x f64) = 16 bytes.
            // MTLSize (3x usize) = 24 bytes.
            // WE MUST DEFINE NSSize/CGSize for this call!

            // Ensure drawable size matches frame (using CGSize)
            let size = sys::CGSize {
                width: frame.width as f64,
                height: frame.height as f64,
            };
            sys::send_1::<(), sys::CGSize>(self.layer, sys::sel(b"setDrawableSize:\0"), size);

            let drawable: Id = sys::send(self.layer, sys::sel(b"nextDrawable\0"));
            if !drawable.is_null() {
                let texture: Id = sys::send(drawable, sys::sel(b"texture\0"));

                let region =
                    sys::MTLRegion::new_2d(0, 0, frame.width as usize, frame.height as usize);
                let bytes = frame.as_bytes().as_ptr() as *const c_void;
                let bytes_per_row = (frame.width as usize) * 4;

                sys::send_4::<(), sys::MTLRegion, usize, *const c_void, usize>(
                    texture,
                    sys::sel(b"replaceRegion:mipmapLevel:withBytes:bytesPerRow:\0"),
                    region,
                    0,
                    bytes,
                    bytes_per_row,
                );

                sys::send::<()>(drawable, sys::sel(b"present\0"));
            }
        }
    }

    pub fn poll_resize(&mut self) -> Option<(u32, u32)> {
        unsafe {
            // View frame relative to window content rect
            // frame includes title bar, so we use contentView bounds for accuracy.
            // Check content view frame? frame includes title bar.
            // We want content size.
            let view: sys::Id = sys::send(self.window.0, sys::sel(b"contentView\0"));
            let bounds: sys::CGRect = sys::send(view, sys::sel(b"bounds\0"));

            let width = bounds.size.width as u32;
            let height = bounds.size.height as u32;

            if width != self.current_width || height != self.current_height {
                self.current_width = width;
                self.current_height = height;
                // Update drawable size immediately to avoid flickering
                self.set_size(width, height);
                return Some((width, height));
            }
        }
        None
    }
}
