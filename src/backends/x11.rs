// src/backends/x11.rs

#![allow(non_snake_case)] // Allow non-snake case for X11 types

// Import logging macros
use log::{debug, info, warn, error, trace};

// Crate-level imports
use crate::glyph::{Color, AttrFlags, NamedColor};
// Updated to use new structs from backends::mod_rs
use crate::backends::{Driver, BackendEvent, CellCoords, TextRunStyle, CellRect}; 

use anyhow::{Context, Result};
use std::ffi::CString;
use std::os::unix::io::RawFd;
use std::ptr;
use std::mem;
use std::collections::HashMap;

// Libc imports
use libc::{c_char, c_int, c_uint};

// X11 library imports
use x11::xlib;
use x11::xft;
use x11::xrender::{XGlyphInfo, XRenderColor};

// --- Constants ---
const DEFAULT_FONT_NAME: &str = "Liberation Mono:size=10";
const MIN_FONT_WIDTH: u32 = 1;
const MIN_FONT_HEIGHT: u32 = 1;

const ANSI_COLOR_COUNT: usize = 16; 
const TOTAL_PREALLOC_COLORS: usize = ANSI_COLOR_COUNT;

const DEFAULT_WINDOW_WIDTH_CHARS: usize = 80;
const DEFAULT_WINDOW_HEIGHT_CHARS: usize = 24;

/// Alpha value for fully opaque XRenderColor.
const XRENDER_ALPHA_OPAQUE: u16 = 0xffff;

/// Helper struct to group 16-bit RGB color components for XRenderColor.
struct Rgb16Components {
    r: u16,
    g: u16,
    b: u16,
}

/// X11 driver implementation for the terminal using Xft for font rendering.
pub struct XDriver {
    display: *mut xlib::Display,
    screen: c_int,
    window: xlib::Window,
    colormap: xlib::Colormap,
    visual: *mut xlib::Visual,
    xft_font: *mut xft::XftFont,
    xft_draw: *mut xft::XftDraw,
    
    xft_ansi_colors: Vec<xft::XftColor>,
    xft_color_cache_rgb: HashMap<(u8, u8, u8), xft::XftColor>,
    
    font_width: u32,
    font_height: u32,
    font_ascent: u32,

    current_pixel_width: u16,
    current_pixel_height: u16,

    wm_delete_window: xlib::Atom,
    protocols_atom: xlib::Atom,
    clear_gc: xlib::GC,
}

impl Driver for XDriver {
    fn new() -> Result<Self> {
        info!("Creating new XDriver");
        let display = unsafe { xlib::XOpenDisplay(ptr::null()) };
        if display.is_null() {
            return Err(anyhow::anyhow!("Failed to open X display: Check DISPLAY environment variable or X server status."));
        }
        debug!("X display opened successfully.");

        let screen = unsafe { xlib::XDefaultScreen(display) };
        let colormap = unsafe { xlib::XDefaultColormap(display, screen) };
        let visual = unsafe { xlib::XDefaultVisual(display, screen) };

        let mut driver = XDriver {
            display,
            screen,
            window: 0,
            colormap,
            visual,
            xft_font: ptr::null_mut(),
            xft_draw: ptr::null_mut(),
            xft_ansi_colors: Vec::with_capacity(TOTAL_PREALLOC_COLORS),
            xft_color_cache_rgb: HashMap::new(),
            font_width: 0,
            font_height: 0,
            font_ascent: 0,
            current_pixel_width: 0,
            current_pixel_height: 0,
            wm_delete_window: 0,
            protocols_atom: 0,
            clear_gc: ptr::null_mut(),
        };

        if let Err(e) = (|| {
            driver.load_font().context("Failed to load font")?;
            
            driver.current_pixel_width = (DEFAULT_WINDOW_WIDTH_CHARS * driver.font_width as usize) as u16;
            driver.current_pixel_height = (DEFAULT_WINDOW_HEIGHT_CHARS * driver.font_height as usize) as u16;

            driver.init_xft_ansi_colors().context("Failed to initialize Xft ANSI colors")?;
            
            let initial_bg_pixel = if !driver.xft_ansi_colors.is_empty() {
                driver.xft_ansi_colors[NamedColor::Black as usize].pixel
            } else {
                warn!("ANSI colors not yet initialized for initial window background, using 0 (black).");
                0 
            };

            driver.create_window(driver.current_pixel_width, driver.current_pixel_height, initial_bg_pixel)
                .context("Failed to create window")?;
            driver.create_gc().context("Failed to create graphics context for clearing")?;

            driver.xft_draw = unsafe { xft::XftDrawCreate(driver.display, driver.window, driver.visual, driver.colormap) };
            if driver.xft_draw.is_null() {
                return Err(anyhow::anyhow!("Failed to create XftDraw object"));
            }
            debug!("XftDraw object created.");

            driver.setup_wm_protocols_and_hints();

            unsafe {
                xlib::XMapWindow(driver.display, driver.window);
                xlib::XFlush(driver.display);
            }
            debug!("Window mapped and flushed.");
            Ok(())
        })() {
            error!("Error during XDriver setup: {:?}", e);
            let _ = driver.cleanup(); 
            return Err(e);
        }

        info!("XDriver initialization complete.");
        Ok(driver)
    }

    fn get_event_fd(&self) -> Option<RawFd> {
        Some(unsafe { xlib::XConnectionNumber(self.display) })
    }

    fn process_events(&mut self) -> Result<Vec<BackendEvent>> {
        let mut backend_events = Vec::new();
        while unsafe { xlib::XPending(self.display) } > 0 {
            let mut xevent: xlib::XEvent = unsafe { mem::zeroed() };
            unsafe { xlib::XNextEvent(self.display, &mut xevent) };

            let event_type = unsafe { xevent.type_ };
            match event_type {
                xlib::Expose => {
                    let xexpose = unsafe { xevent.expose };
                    if xexpose.count == 0 {
                        debug!("XEvent: Expose (x:{}, y:{}, w:{}, h:{}) - redraw will be handled by renderer", 
                               xexpose.x, xexpose.y, xexpose.width, xexpose.height);
                    }
                }
                xlib::ConfigureNotify => {
                    let xconfigure = unsafe { xevent.configure };
                    if self.current_pixel_width != xconfigure.width as u16 || self.current_pixel_height != xconfigure.height as u16 {
                        debug!(
                            "XEvent: ConfigureNotify (resize from {}x{} to {}x{})",
                            self.current_pixel_width, self.current_pixel_height,
                            xconfigure.width, xconfigure.height
                        );
                        self.current_pixel_width = xconfigure.width as u16;
                        self.current_pixel_height = xconfigure.height as u16;
                        backend_events.push(BackendEvent::Resize {
                            width_px: self.current_pixel_width,
                            height_px: self.current_pixel_height,
                        });
                    }
                }
                xlib::KeyPress => {
                    let mut keysym: xlib::KeySym = 0;
                    let mut key_text_buffer: [u8; 32] = [0; 32];
                    let count = unsafe {
                        xlib::XLookupString(
                            &mut xevent.key,
                            key_text_buffer.as_mut_ptr() as *mut c_char,
                            key_text_buffer.len() as c_int,
                            &mut keysym,
                            ptr::null_mut(),
                        )
                    };
                    let text = if count > 0 {
                        String::from_utf8_lossy(&key_text_buffer[0..count as usize]).to_string()
                    } else {
                        String::new()
                    };
                    debug!("XEvent: KeyPress (keysym: {:#X}, text: '{}')", keysym, text);
                    backend_events.push(BackendEvent::Key {
                        keysym: keysym as u32,
                        text,
                    });
                }
                xlib::ClientMessage => {
                    let xclient = unsafe { xevent.client_message };
                    if xclient.message_type == self.protocols_atom && xclient.data.get_long(0) as xlib::Atom == self.wm_delete_window {
                        info!("XEvent: WM_DELETE_WINDOW received from window manager.");
                        backend_events.push(BackendEvent::CloseRequested);
                    } else {
                        trace!("XEvent: Ignored ClientMessage (type: {})", xclient.message_type);
                    }
                }
                xlib::FocusIn => {
                    debug!("XEvent: FocusIn received.");
                    backend_events.push(BackendEvent::FocusGained);
                }
                xlib::FocusOut => {
                    debug!("XEvent: FocusOut received.");
                    backend_events.push(BackendEvent::FocusLost);
                }
                _ => {
                    trace!("XEvent: Ignored (type: {})", event_type);
                }
            }
        }
        Ok(backend_events)
    }

    fn get_font_dimensions(&self) -> (usize, usize) {
        (self.font_width as usize, self.font_height as usize)
    }

    fn get_display_dimensions_pixels(&self) -> (u16, u16) {
        (self.current_pixel_width, self.current_pixel_height)
    }

    fn clear_all(&mut self, bg: Color) -> Result<()> {
        if matches!(bg, Color::Default) {
            error!("XDriver::clear_all received Color::Default. This is a bug in the Renderer.");
             let xft_bg_color = self.cached_rgb_to_xft_color(0,0,0) // Fallback to black
                .context("Failed to resolve fallback black color for clear_all")?;
             unsafe {
                xlib::XSetForeground(self.display, self.clear_gc, xft_bg_color.pixel);
                xlib::XFillRectangle(
                    self.display, self.window, self.clear_gc,
                    0, 0, self.current_pixel_width as u32, self.current_pixel_height as u32,
                );
            }
            // It's better to panic or return error to highlight the logic issue upstream
            return Err(anyhow::anyhow!("XDriver::clear_all received Color::Default. Renderer should resolve defaults."));
        }

        let xft_bg_color = self.resolve_concrete_xft_color(bg)
            .context("Failed to resolve background color for clear_all")?;
        
        unsafe {
            xlib::XSetForeground(self.display, self.clear_gc, xft_bg_color.pixel);
            xlib::XFillRectangle(
                self.display, self.window, self.clear_gc,
                0, 0, self.current_pixel_width as u32, self.current_pixel_height as u32,
            );
        }
        trace!("Window cleared with color pixel: {}", xft_bg_color.pixel);
        Ok(())
    }

    fn draw_text_run(
        &mut self,
        coords: CellCoords,
        text: &str,
        style: TextRunStyle,
    ) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        if matches!(style.fg, Color::Default) || matches!(style.bg, Color::Default) {
            error!("XDriver::draw_text_run received Color::Default in style. This is a bug in the Renderer.");
            return Err(anyhow::anyhow!("XDriver::draw_text_run received Color::Default. Renderer should resolve defaults."));
        }

        let x_pixel = (coords.x * self.font_width as usize) as c_int;
        let y_pixel = (coords.y * self.font_height as usize) as c_int;
        let run_pixel_width = text.chars().count() * self.font_width as usize;

        let xft_fg = self.resolve_concrete_xft_color(style.fg).context("Failed to resolve foreground color for text run")?;
        let xft_bg = self.resolve_concrete_xft_color(style.bg).context("Failed to resolve background color for text run")?;

        unsafe {
            xft::XftDrawRect(
                self.xft_draw, &xft_bg, x_pixel, y_pixel,
                run_pixel_width as u32, self.font_height,
            );
        }

        let c_text = CString::new(text).context("Failed to convert text to CString for Xft")?;
        let baseline_y_pixel = y_pixel + self.font_ascent as c_int;
        unsafe {
            xft::XftDrawStringUtf8(
                self.xft_draw, &xft_fg, self.xft_font,
                x_pixel, baseline_y_pixel,
                c_text.as_ptr() as *const u8, c_text.as_bytes().len() as c_int,
            );
        }
        
        if style.flags.contains(AttrFlags::UNDERLINE) {
            let underline_y = y_pixel + self.font_height as c_int - 2; 
            unsafe {
                xft::XftDrawRect(self.xft_draw, &xft_fg, x_pixel, underline_y, run_pixel_width as u32, 1);
            }
        }
        if style.flags.contains(AttrFlags::STRIKETHROUGH) {
            let strikethrough_y = y_pixel + (self.font_ascent / 2) as c_int;
            unsafe {
                xft::XftDrawRect(self.xft_draw, &xft_fg, x_pixel, strikethrough_y, run_pixel_width as u32, 1);
            }
        }
        Ok(())
    }

    fn fill_rect(
        &mut self,
        rect: CellRect,
        color: Color,
    ) -> Result<()> {
        if rect.width == 0 || rect.height == 0 {
            return Ok(());
        }
        if matches!(color, Color::Default) {
            error!("XDriver::fill_rect received Color::Default. This is a bug in the Renderer.");
             return Err(anyhow::anyhow!("XDriver::fill_rect received Color::Default. Renderer should resolve defaults."));
        }

        let x_pixel = (rect.x * self.font_width as usize) as c_int;
        let y_pixel = (rect.y * self.font_height as usize) as c_int;
        let rect_pixel_width = (rect.width * self.font_width as usize) as u32;
        let rect_pixel_height = (rect.height * self.font_height as usize) as u32;

        let xft_fill_color = self.resolve_concrete_xft_color(color).context("Failed to resolve color for fill_rect")?;

        unsafe {
            xft::XftDrawRect(
                self.xft_draw, &xft_fill_color, x_pixel, y_pixel,
                rect_pixel_width, rect_pixel_height,
            );
        }
        Ok(())
    }

    fn present(&mut self) -> Result<()> {
        unsafe {
            xlib::XFlush(self.display);
        }
        trace!("XFlush called to present frame.");
        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        info!("Cleaning up XDriver resources...");
        unsafe {
            if !self.xft_draw.is_null() {
                trace!("Destroying XftDraw object.");
                xft::XftDrawDestroy(self.xft_draw);
                self.xft_draw = ptr::null_mut();
            }
            if !self.xft_font.is_null() {
                trace!("Closing XftFont.");
                xft::XftFontClose(self.display, self.xft_font);
                self.xft_font = ptr::null_mut();
            }
            
            trace!("Freeing {} preallocated ANSI XftColors.", self.xft_ansi_colors.len());
            for color_ptr in self.xft_ansi_colors.iter_mut() {
                xft::XftColorFree(self.display, self.visual, self.colormap, color_ptr);
            }
            self.xft_ansi_colors.clear();

            trace!("Freeing {} cached RGB XftColors.", self.xft_color_cache_rgb.len());
            for (_, mut cached_color) in self.xft_color_cache_rgb.drain() {
                 xft::XftColorFree(self.display, self.visual, self.colormap, &mut cached_color);
            }

            if !self.clear_gc.is_null() {
                trace!("Freeing clear GC.");
                xlib::XFreeGC(self.display, self.clear_gc);
                self.clear_gc = ptr::null_mut();
            }
            if self.window != 0 {
                trace!("Destroying X window (ID: {}).", self.window);
                xlib::XDestroyWindow(self.display, self.window);
                self.window = 0;
            }
            if !self.display.is_null() {
                trace!("Closing X display connection.");
                xlib::XCloseDisplay(self.display);
                self.display = ptr::null_mut();
            }
        }
        info!("XDriver cleanup complete.");
        Ok(())
    }
}

impl XDriver {
    fn load_font(&mut self) -> Result<()> {
        debug!("Loading font: {}", DEFAULT_FONT_NAME);
        let font_name_cstr = CString::new(DEFAULT_FONT_NAME)
            .context("Failed to create CString for font name")?;
        
        self.xft_font = unsafe { xft::XftFontOpenName(self.display, self.screen, font_name_cstr.as_ptr()) };
        if self.xft_font.is_null() {
            return Err(anyhow::anyhow!(
                "XftFontOpenName failed for font: '{}'. Ensure font is installed and accessible.",
                DEFAULT_FONT_NAME
            ));
        }
        debug!("Font '{}' loaded successfully.", DEFAULT_FONT_NAME);

        let font_info_ptr = self.xft_font;
        self.font_height = unsafe { ((*font_info_ptr).ascent + (*font_info_ptr).descent) as u32 };
        self.font_ascent = unsafe { (*font_info_ptr).ascent as u32 };
        
        let mut extents: XGlyphInfo = unsafe { mem::zeroed() };
        let sample_char_cstr = CString::new("M").expect("CString::new for 'M' failed.");
        unsafe {
            xft::XftTextExtentsUtf8(
                self.display, self.xft_font,
                sample_char_cstr.as_ptr() as *const u8,
                sample_char_cstr.as_bytes().len() as c_int,
                &mut extents,
            );
        }
        self.font_width = extents.xOff as u32;

        if self.font_width < MIN_FONT_WIDTH || self.font_height < MIN_FONT_HEIGHT {
            return Err(anyhow::anyhow!(
                "Font dimensions (W:{}, H:{}) below minimum (W:{}, H:{}).",
                self.font_width, self.font_height, MIN_FONT_WIDTH, MIN_FONT_HEIGHT
            ));
        }
        info!("Font metrics: Width={}, Height={}, Ascent={}", self.font_width, self.font_height, self.font_ascent);
        Ok(())
    }

    fn init_xft_ansi_colors(&mut self) -> Result<()> {
        debug!("Initializing {} preallocated ANSI Xft colors.", ANSI_COLOR_COUNT);
        self.xft_ansi_colors.resize_with(ANSI_COLOR_COUNT, || unsafe { mem::zeroed() });

        for i in 0..ANSI_COLOR_COUNT {
            let named_color = NamedColor::from_index(i as u8);
            let (r_u8, g_u8, b_u8) = match named_color {
                NamedColor::Black         => (0x00, 0x00, 0x00), NamedColor::Red           => (0xCD, 0x00, 0x00),
                NamedColor::Green         => (0x00, 0xCD, 0x00), NamedColor::Yellow        => (0xCD, 0xCD, 0x00),
                NamedColor::Blue          => (0x00, 0x00, 0xEE), NamedColor::Magenta       => (0xCD, 0x00, 0xCD),
                NamedColor::Cyan          => (0x00, 0xCD, 0xCD), NamedColor::White         => (0xE5, 0xE5, 0xE5),
                NamedColor::BrightBlack   => (0x7F, 0x7F, 0x7F), NamedColor::BrightRed     => (0xFF, 0x00, 0x00),
                NamedColor::BrightGreen   => (0x00, 0xFF, 0x00), NamedColor::BrightYellow  => (0xFF, 0xFF, 0x00),
                NamedColor::BrightBlue    => (0x5C, 0x5C, 0xFF), NamedColor::BrightMagenta => (0xFF, 0x00, 0xFF),
                NamedColor::BrightCyan    => (0x00, 0xFF, 0xFF), NamedColor::BrightWhite   => (0xFF, 0xFF, 0xFF),
            };
            let color_components = Rgb16Components {
                r: ((r_u8 as u16) << 8) | (r_u8 as u16),
                g: ((g_u8 as u16) << 8) | (g_u8 as u16),
                b: ((b_u8 as u16) << 8) | (b_u8 as u16),
            };
            // Call the corrected method signature
            XDriver::alloc_specific_xft_color_into_slice(self, i, color_components, &format!("ANSI {} ({:?})", i, named_color))?;
        }
        
        debug!("Preallocated ANSI Xft colors initialized.");
        Ok(())
    }
    
    /// Helper to allocate a specific XftColor into the provided slice at `index`.
    /// Changed to take `&self` instead of `&mut self`.
    fn alloc_specific_xft_color_into_slice(
        &mut self, // Changed from &mut self
        index: usize, 
        color_comps: Rgb16Components, 
        name_for_log: &str,
    ) -> Result<()> {
        let render_color = XRenderColor { 
            red: color_comps.r, green: color_comps.g, blue: color_comps.b, 
            alpha: XRENDER_ALPHA_OPAQUE 
        };
        // self.display, self.visual, self.colormap are accessed via &self
        if unsafe { xft::XftColorAllocValue(self.display, self.visual, self.colormap, &render_color, &mut self.xft_ansi_colors[index]) } == 0 {
            return Err(anyhow::anyhow!("XftColorAllocValue failed for {}", name_for_log));
        }
        trace!("Allocated XftColor for {} (idx {}, pixel: {})", name_for_log, index, &self.xft_ansi_colors[index].pixel);
        Ok(())
    }

    fn create_window(&mut self, pixel_width: u16, pixel_height: u16, bg_pixel_val: xlib::Atom) -> Result<()> {
        unsafe {
            let root_window = xlib::XRootWindow(self.display, self.screen);
            let border_width = 0;

            let mut attributes: xlib::XSetWindowAttributes = mem::zeroed();
            attributes.colormap = self.colormap;
            attributes.background_pixel = bg_pixel_val; 
            attributes.border_pixel = bg_pixel_val;
            attributes.event_mask = xlib::ExposureMask | xlib::KeyPressMask |
                xlib::StructureNotifyMask | xlib::FocusChangeMask;

            self.window = xlib::XCreateWindow(
                self.display, root_window, 0, 0,
                pixel_width as c_uint, pixel_height as c_uint,
                border_width, xlib::XDefaultDepth(self.display, self.screen),
                xlib::InputOutput as c_uint, self.visual,
                xlib::CWColormap | xlib::CWBackPixel | xlib::CWBorderPixel | xlib::CWEventMask,
                &mut attributes,
            );
        }
        if self.window == 0 {
            return Err(anyhow::anyhow!("XCreateWindow failed."));
        }
        debug!("X window created (ID: {}), initial size: {}x{}px", self.window, pixel_width, pixel_height);
        self.current_pixel_width = pixel_width;
        self.current_pixel_height = pixel_height;
        Ok(())
    }

    fn create_gc(&mut self) -> Result<()> {
        let gc_values: xlib::XGCValues = unsafe { mem::zeroed() };
        self.clear_gc = unsafe { xlib::XCreateGC(self.display, self.window, 0, &gc_values as *const _ as *mut _) };
        if self.clear_gc.is_null() {
            return Err(anyhow::anyhow!("XCreateGC failed."));
        }
        debug!("Graphics Context (GC) for clearing created.");
        Ok(())
    }

    fn setup_wm_protocols_and_hints(&mut self) {
        unsafe {
            self.wm_delete_window = xlib::XInternAtom(self.display, b"WM_DELETE_WINDOW\0".as_ptr() as *mut _, xlib::False);
            self.protocols_atom = xlib::XInternAtom(self.display, b"WM_PROTOCOLS\0".as_ptr() as *mut _, xlib::False);
            
            if self.wm_delete_window != 0 && self.protocols_atom != 0 {
                 xlib::XSetWMProtocols(self.display, self.window, [self.wm_delete_window].as_mut_ptr(), 1);
                 debug!("WM_PROTOCOLS (WM_DELETE_WINDOW) registered.");
            } else {
                warn!("Failed to get WM_DELETE_WINDOW or WM_PROTOCOLS atom.");
            }

            let title_cstr = CString::new("myterm").expect("CString::new for 'myterm' failed.");
            xlib::XStoreName(self.display, self.window, title_cstr.as_ptr() as *mut c_char);
            
            let net_wm_name_atom = xlib::XInternAtom(self.display, b"_NET_WM_NAME\0".as_ptr() as *mut _, xlib::False);
            let utf8_string_atom = xlib::XInternAtom(self.display, b"UTF8_STRING\0".as_ptr() as *mut _, xlib::False);
            if net_wm_name_atom != 0 && utf8_string_atom != 0 {
                 xlib::XChangeProperty(
                    self.display, self.window, net_wm_name_atom, utf8_string_atom, 8,
                    xlib::PropModeReplace, title_cstr.as_ptr() as *const u8, 
                    title_cstr.as_bytes().len() as c_int
                );
                 debug!("Window title set via XStoreName and _NET_WM_NAME.");
            } else {
                debug!("Window title set via XStoreName only.");
            }

            let mut size_hints: xlib::XSizeHints = mem::zeroed();
            size_hints.flags = xlib::PResizeInc | xlib::PMinSize;
            size_hints.width_inc = self.font_width as c_int;
            size_hints.height_inc = self.font_height as c_int;
            size_hints.min_width = self.font_width as c_int;
            size_hints.min_height = self.font_height as c_int;
            xlib::XSetWMNormalHints(self.display, self.window, &mut size_hints);
            debug!("WM size hints set.");
        }
    }
    
    fn resolve_concrete_xft_color(&mut self, color: Color) -> Result<xft::XftColor> {
        match color {
            Color::Default => {
                error!("XDriver::resolve_concrete_xft_color received Color::Default. This is a bug in the Renderer.");
                panic!("XDriver received Color::Default. Renderer should resolve all defaults before passing to driver.");
            }
            Color::Named(named_color) => {
                Ok(self.xft_ansi_colors[named_color as u8 as usize])
            }
            Color::Indexed(idx) => {
                if (idx as usize) < ANSI_COLOR_COUNT {
                     Ok(self.xft_ansi_colors[idx as usize])
                } else {
                    let (r,g,b) = convert_indexed_to_rgb_approximation(idx);
                    warn!("XDriver: Approximating Indexed({}) to RGB({},{},{})", idx, r,g,b);
                    self.cached_rgb_to_xft_color(r,g,b)
                }
            }
            Color::Rgb(r, g, b) => {
                self.cached_rgb_to_xft_color(r,g,b)
            }
        }
    }

    fn cached_rgb_to_xft_color(&mut self, r_u8: u8, g_u8: u8, b_u8: u8) -> Result<xft::XftColor> {
        if let Some(cached_color) = self.xft_color_cache_rgb.get(&(r_u8, g_u8, b_u8)) {
            return Ok(*cached_color);
        }

        let color_components = Rgb16Components {
            r: ((r_u8 as u16) << 8) | (r_u8 as u16),
            g: ((g_u8 as u16) << 8) | (g_u8 as u16),
            b: ((b_u8 as u16) << 8) | (b_u8 as u16),
        };

        let render_color = XRenderColor { 
            red: color_components.r, green: color_components.g, blue: color_components.b, 
            alpha: XRENDER_ALPHA_OPAQUE
        };
        let mut new_xft_color: xft::XftColor = unsafe { mem::zeroed() };
        
        if unsafe { xft::XftColorAllocValue(self.display, self.visual, self.colormap, &render_color, &mut new_xft_color) } == 0 {
            Err(anyhow::anyhow!("XftColorAllocValue failed for RGB({},{},{})", r_u8, g_u8, b_u8))
        } else {
            self.xft_color_cache_rgb.insert((r_u8, g_u8, b_u8), new_xft_color);
            Ok(new_xft_color)
        }
    }
}

fn convert_indexed_to_rgb_approximation(idx: u8) -> (u8,u8,u8) {
    if idx < 16 { 
        let named = NamedColor::from_index(idx);
         match named {
            NamedColor::Black => (0,0,0), NamedColor::Red => (205,0,0), NamedColor::Green => (0,205,0),
            NamedColor::Yellow => (205,205,0), NamedColor::Blue => (0,0,238), NamedColor::Magenta => (205,0,205),
            NamedColor::Cyan => (0,205,205), NamedColor::White => (229,229,229),
            NamedColor::BrightBlack => (127,127,127), NamedColor::BrightRed => (255,0,0),
            NamedColor::BrightGreen => (0,255,0), NamedColor::BrightYellow => (255,255,0),
            NamedColor::BrightBlue => (92,92,255), NamedColor::BrightMagenta => (255,0,255),
            NamedColor::BrightCyan => (0,255,255), NamedColor::BrightWhite => (255,255,255),
        }
    } else if idx >= 232 { 
        let level = (idx - 232) * 10 + 8;
        (level, level, level)
    } else if idx >= 16 { 
        let i = idx - 16;
        let r_idx = (i / 36) % 6;
        let g_idx = (i / 6) % 6;
        let b_idx = i % 6;
        let r_val = if r_idx == 0 {0} else {r_idx*40+55};
        let g_val = if g_idx == 0 {0} else {g_idx*40+55};
        let b_val = if b_idx == 0 {0} else {b_idx*40+55};
        (r_val, g_val, b_val)
    } else {
        (0,0,0) 
    }
}

impl Drop for XDriver {
    fn drop(&mut self) {
        info!("Dropping XDriver instance, performing cleanup.");
        if let Err(e) = self.cleanup() {
            error!("Error during XDriver cleanup in drop: {}", e);
        }
    }
}
