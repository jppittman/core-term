// src/platform/backends/x11/graphics.rs
#![allow(non_snake_case)] // Allow non-snake case for X11 types

use super::connection::Connection;
use crate::color::{Color, NamedColor};
use crate::glyph::AttrFlags;
use crate::platform::backends::{CellCoords, CellRect, TextRunStyle};

use anyhow::{anyhow, Context, Result}; // Combined anyhow
use log::{debug, error, info, trace, warn};
use std::collections::HashMap;
use std::ffi::CString;
use std::mem;
use std::ptr;

// X11 library imports
use libc::c_int;
use x11::xrender::{XGlyphInfo, XRenderColor};
use x11::{xft, xlib};

// --- RAII Wrappers for X11 Resources ---

/// Wraps an `XftFont` pointer to ensure it's closed via `XftFontClose` on drop.
#[derive(Debug)]
pub(super) struct SafeXftFont { // Made pub(super)
    ptr: *mut xft::XftFont,
    display: *mut xlib::Display, // Needed for XftFontClose
}

impl SafeXftFont {
    fn new(font_ptr: *mut xft::XftFont, display_ptr: *mut xlib::Display) -> Self {
        Self {
            ptr: font_ptr,
            display: display_ptr,
        }
    }

    #[inline]
    fn raw(&self) -> *mut xft::XftFont {
        self.ptr
    }

    // Helper to access fields for metrics, assuming ptr is valid.
    // Unsafe because it dereferences the raw pointer.
    unsafe fn ascent(&self) -> c_int {
        if self.ptr.is_null() {
            panic!("Attempted to access ascent on a null XftFont pointer.");
        }
        (*self.ptr).ascent
    }

    unsafe fn descent(&self) -> c_int {
        if self.ptr.is_null() {
            panic!("Attempted to access descent on a null XftFont pointer.");
        }
        (*self.ptr).descent
    }
}

impl Drop for SafeXftFont {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if self.display.is_null() {
                warn!("SafeXftFont::drop called with a null display pointer. Cannot close font: {:p}. This is a bug.", self.ptr);
                return;
            }
            trace!("Closing XftFont via SafeXftFont drop: {:p}", self.ptr);
            unsafe { xft::XftFontClose(self.display, self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

/// Wraps an `XftDraw` pointer to ensure it's destroyed via `XftDrawDestroy` on drop.
#[derive(Debug)]
pub(super) struct SafeXftDraw { // Made pub(super)
    ptr: *mut xft::XftDraw,
}

impl SafeXftDraw {
    fn new(draw_ptr: *mut xft::XftDraw) -> Self {
        Self { ptr: draw_ptr }
    }

    #[inline]
    fn raw(&self) -> *mut xft::XftDraw {
        self.ptr
    }
}

impl Drop for SafeXftDraw {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            trace!("Destroying XftDraw via SafeXftDraw drop: {:p}", self.ptr);
            unsafe { xft::XftDrawDestroy(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

/// Wraps an X11 `GC` (Graphics Context) to ensure it's freed via `XFreeGC` on drop.
#[derive(Debug)]
pub(super) struct SafeGc { // Made pub(super)
    gc: xlib::GC, // This is an XID (typically ulong), not a pointer.
    display: *mut xlib::Display,
    is_valid: bool, // To track if GC is valid and needs freeing.
}

impl SafeGc {
    fn new(gc: xlib::GC, display_ptr: *mut xlib::Display) -> Self {
        Self {
            gc,
            display: display_ptr,
            is_valid: !gc.is_null(), // Assuming null GC is invalid.
        }
    }

    #[inline]
    fn raw(&self) -> xlib::GC {
        if self.is_valid { self.gc } else { ptr::null_mut() } // Or handle error
    }
}

impl Drop for SafeGc {
    fn drop(&mut self) {
        if self.is_valid && !self.gc.is_null() {
            if self.display.is_null() {
                warn!("SafeGc::drop called with a null display pointer. Cannot free GC: {:p}. This is a bug.", self.gc);
                return;
            }
            trace!("Freeing GC via SafeGc drop: {:p}", self.gc);
            unsafe { xlib::XFreeGC(self.display, self.gc) };
            self.is_valid = false;
            // self.gc cannot be set to null_mut if it's not a pointer.
        }
    }
}

/// Wraps an `XftColor` structure to ensure it's freed via `XftColorFree` on drop.
#[derive(Debug, Clone)] // Removed Copy
pub(super) struct SafeXftColor { // Made pub(super)
    color: xft::XftColor,
    display: *mut xlib::Display,
    visual: *mut xlib::Visual,
    colormap: xlib::Colormap,
    is_allocated: bool,
}

impl SafeXftColor {
    fn new(
        xft_color_data: xft::XftColor,
        display_ptr: *mut xlib::Display,
        visual_ptr: *mut xlib::Visual,
        colormap_val: xlib::Colormap,
        is_allocated: bool,
    ) -> Self {
        Self {
            color: xft_color_data,
            display: display_ptr,
            visual: visual_ptr,
            colormap: colormap_val,
            is_allocated,
        }
    }

    #[inline]
    fn cloned_color(&self) -> xft::XftColor { // Changed from raw() to cloned_color()
        self.color // xft::XftColor is Copy
    }

    fn raw_mut_for_free(&mut self) -> *mut xft::XftColor {
        &mut self.color
    }
}

impl Drop for SafeXftColor {
    fn drop(&mut self) {
        if self.is_allocated {
            if self.display.is_null() || self.visual.is_null() || self.colormap == 0 {
                warn!("SafeXftColor::drop called with null display/visual or zero colormap. Color pixel: {}. This may indicate a bug or premature cleanup of Connection.", self.color.pixel);
                return;
            }
            trace!(
                "Freeing XftColor via SafeXftColor drop, pixel: {}",
                self.color.pixel
            );
            unsafe {
                xft::XftColorFree(
                    self.display,
                    self.visual,
                    self.colormap,
                    self.raw_mut_for_free(),
                )
            };
            self.is_allocated = false;
        }
    }
}

// --- Constants ---

/// Default font name and size to use if not otherwise specified.
const DEFAULT_FONT_NAME: &str = "Inconsolata:size=10";
/// Minimum acceptable width for a loaded font's character cell in pixels.
const MIN_FONT_WIDTH: u32 = 1;
/// Minimum acceptable height for a loaded font's character cell in pixels.
const MIN_FONT_HEIGHT: u32 = 1;

/// Number of standard ANSI colors (0-15) to preallocate.
const ANSI_COLOR_COUNT: usize = 16;
/// Alpha value for fully opaque colors in XRender.
const XRENDER_ALPHA_OPAQUE: u16 = 0xffff;

/// Helper struct for 16-bit RGB color components, used with `XRenderColor`.
#[derive(Debug, Clone, Copy)]
struct Rgb16Components {
    r: u16,
    g: u16,
    b: u16,
}

/// Holds data from the first stage of graphics initialization, before the window ID is known.
/// This data includes loaded font resources, calculated metrics, pre-allocated ANSI colors,
/// and the determined default background pixel value. It's passed to the second stage
/// of `Graphics::new` once the window is created.
pub(super) struct PreGraphicsData {
    pub(super) xft_font: SafeXftFont, // Ownership of the XftFont
    pub(super) font_width_px: u32,
    pub(super) font_height_px: u32,
    pub(super) font_ascent_px: u32,
    pub(super) xft_ansi_colors: Vec<SafeXftColor>, // Owns the XftColors
    pub(super) default_bg_pixel_value: xlib::Atom, // Typically XftColor.pixel (ulong)
}

/// Manages graphics resources for X11 rendering, including fonts, colors,
/// and drawing primitives.
///
/// This struct encapsulates Xft resources for font rendering, color allocation
/// (pre-allocated ANSI colors and a cache for other RGB colors), and a Graphics Context (GC)
/// for simple drawing operations. It's initialized in two stages to resolve
/// dependencies between font metrics, initial background color, and window creation.
///
/// Resource cleanup is handled by the `cleanup` method, which should be called
/// explicitly by `XDriver`. The `Drop` trait provides a fallback log message if
/// cleanup was not performed.
#[derive(Debug)]
pub struct Graphics {
    xft_font: SafeXftFont, // Owns the XftFont resource
    xft_draw: SafeXftDraw, // Owns the XftDraw resource
    xft_ansi_colors: Vec<SafeXftColor>, // Owns the pre-allocated ANSI XftColors
    xft_color_cache_rgb: HashMap<(u8, u8, u8), SafeXftColor>, // Owns cached XftColors
    font_width_px: u32,
    font_height_px: u32,
    font_ascent_px: u32,
    clear_gc: SafeGc, // Owns the Graphics Context
}

impl Graphics {
    /// Performs the first stage of graphics initialization.
    ///
    /// This stage does not require a window ID. It loads the default font,
    /// calculates font metrics (width, height, ascent), pre-allocates the
    /// standard 16 ANSI colors using Xft, and determines the pixel value
    /// for the default background color (black).
    ///
    /// # Arguments
    /// * `connection`: A reference to the active X11 `Connection`.
    ///
    /// # Returns
    /// * `Ok(PreGraphicsData)`: Contains the loaded font, metrics, ANSI colors, and default background pixel.
    /// * `Err(anyhow::Error)`: If font loading or color allocation fails.
    pub(super) fn load_font_and_colors(connection: &Connection) -> Result<PreGraphicsData> {
        info!("Graphics: Loading font and pre-allocating colors.");
        let display = connection.display();
        let screen = connection.screen();

        // 1. Load Font & Metrics
        debug!("Loading font: {}", DEFAULT_FONT_NAME);
        let font_name_cstr =
            CString::new(DEFAULT_FONT_NAME).context("Failed to create CString for font name")?;

        // SAFETY: Xlib/Xft FFI call. `display` and `screen` must be valid.
        let xft_font_raw_ptr =
            unsafe { xft::XftFontOpenName(display, screen, font_name_cstr.as_ptr()) };
        if xft_font_raw_ptr.is_null() {
            return Err(anyhow!(
                "XftFontOpenName failed for font: '{}'. Ensure font is installed and accessible.",
                DEFAULT_FONT_NAME
            ));
        }
        // Wrap immediately in SafeXftFont. If later stages fail, its Drop will handle cleanup.
        let xft_font = SafeXftFont::new(xft_font_raw_ptr, display);
        debug!(
            "Font '{}' loaded successfully: {:p}",
            DEFAULT_FONT_NAME,
            xft_font.raw()
        );

        // SAFETY: Accessing fields of a valid `xft_font` pointer via helper methods.
        let font_height_px = unsafe { (xft_font.ascent() + xft_font.descent()) as u32 };
        let font_ascent_px = unsafe { xft_font.ascent() as u32 };

        let mut extents: XGlyphInfo = unsafe { mem::zeroed() };
        let sample_char_cstr = CString::new("M").expect("CString::new for 'M' should not fail.");
        // SAFETY: Xlib/Xft FFI call. `display` and `xft_font.raw()` must be valid.
        unsafe {
            xft::XftTextExtentsUtf8(
                display,
                xft_font.raw(),
                sample_char_cstr.as_ptr() as *const u8,
                sample_char_cstr.as_bytes().len() as c_int,
                &mut extents,
            );
        }
        let font_width_px = extents.xOff as u32;

        if font_width_px < MIN_FONT_WIDTH || font_height_px < MIN_FONT_HEIGHT {
            // xft_font's Drop will be called automatically when it goes out of scope here.
            return Err(anyhow!(
                "Font dimensions (W:{}, H:{}) are below minimum requirements (W:{}, H:{}).",
                font_width_px,
                font_height_px,
                MIN_FONT_WIDTH,
                MIN_FONT_HEIGHT
            ));
        }
        info!(
            "Font metrics determined: Width={}, Height={}, Ascent={}",
            font_width_px, font_height_px, font_ascent_px
        );

        // 2. Initialize ANSI Colors
        let mut xft_ansi_colors = Vec::with_capacity(ANSI_COLOR_COUNT);
        let visual = connection.visual();
        let colormap = connection.colormap();

        for i in 0..ANSI_COLOR_COUNT {
            let named_color_enum = NamedColor::from_index(i as u8);
            let rgb_color = named_color_enum.to_rgb_color();
            let (r_u8, g_u8, b_u8) = match rgb_color {
                Color::Rgb(r, g, b) => (r, g, b),
                _ => {
                    warn!(
                        "NamedColor::to_rgb_color for index {} did not return Color::Rgb. Defaulting to black.",
                        i
                    );
                    (0, 0, 0)
                }
            };
            let color_components = Rgb16Components {
                r: ((r_u8 as u16) << 8) | (r_u8 as u16),
                g: ((g_u8 as u16) << 8) | (g_u8 as u16),
                b: ((b_u8 as u16) << 8) | (b_u8 as u16),
            };
            let render_color = XRenderColor {
                red: color_components.r,
                green: color_components.g,
                blue: color_components.b,
                alpha: XRENDER_ALPHA_OPAQUE,
            };

            let mut current_xft_color_data: xft::XftColor = unsafe { mem::zeroed() };
            // SAFETY: FFI call. display, visual, colormap must be valid.
            let success = unsafe {
                xft::XftColorAllocValue(
                    display,
                    visual,
                    colormap,
                    &render_color,
                    &mut current_xft_color_data,
                ) != 0 // Returns 0 on failure, non-zero on success.
            };

            if success {
                xft_ansi_colors.push(SafeXftColor::new(
                    current_xft_color_data,
                    display,
                    visual,
                    colormap,
                    true, // Mark as allocated
                ));
            } else {
                // If allocation fails, xft_font's Drop will run, and any
                // SafeXftColors already in xft_ansi_colors will also be dropped correctly.
                return Err(anyhow!(
                    "XftColorAllocValue failed for ANSI color index {}",
                    i
                ));
            }
        }
        debug!("Preallocated ANSI Xft colors initialized.");

        // Determine default background pixel value from the pre-allocated black.
        // Accessing .color.pixel directly on SafeXftColor via cloned_color().
        let default_bg_pixel_value = xft_ansi_colors[NamedColor::Black as usize].cloned_color().pixel;
        info!(
            "Default background pixel value (from Black ANSI color): {}",
            default_bg_pixel_value
        );

        Ok(PreGraphicsData {
            xft_font, // Now a SafeXftFont
            font_width_px,
            font_height_px,
            font_ascent_px,
            xft_ansi_colors, // Now a Vec<SafeXftColor>
            default_bg_pixel_value,
        })
    }

    /// Finalizes graphics initialization using pre-loaded font/color data and a window ID.
    ///
    /// This second stage creates resources that depend on a specific window, such as
    /// the `XftDraw` object and a Graphics Context (GC) for clearing.
    ///
    /// # Arguments
    /// * `connection`: A reference to the active X11 `Connection`.
    /// * `window_id`: The ID of the X11 window to associate with drawing operations.
    /// * `pre_data`: The `PreGraphicsData` containing font and color resources from the first stage.
    ///
    /// # Returns
    /// * `Ok(Graphics)`: A fully initialized `Graphics` instance.
    /// * `Err(anyhow::Error)`: If creating `XftDraw` or GC fails. Resources from `pre_data`
    ///   are cleaned up internally in case of error.
    pub(super) fn new(
        connection: &Connection,
        window_id: xlib::Window,
        pre_data: PreGraphicsData, // Contains SafeXftFont and Vec<SafeXftColor>
    ) -> Result<Self> {
        info!(
            "Graphics: Finalizing initialization with window ID: {}",
            window_id
        );
        let display = connection.display();
        let visual = connection.visual();
        let colormap = connection.colormap();

        // 1. Create XftDraw object for the window.
        // SAFETY: FFI call. display, window_id, visual, colormap must be valid.
        let xft_draw_raw_ptr =
            unsafe { xft::XftDrawCreate(display, window_id, visual, colormap) };

        if xft_draw_raw_ptr.is_null() {
            // If this fails, pre_data (including its SafeXftFont and Vec<SafeXftColor>)
            // will be dropped automatically, cleaning up those resources.
            return Err(anyhow!(
                "Failed to create XftDraw object for window ID {}",
                window_id
            ));
        }
        let xft_draw = SafeXftDraw::new(xft_draw_raw_ptr);
        debug!("XftDraw object created: {:p}", xft_draw.raw());

        // 2. Create Graphics Context (GC).
        let gc_values: xlib::XGCValues = unsafe { mem::zeroed() };
        // SAFETY: FFI call. display and window_id must be valid.
        let clear_gc_raw_ptr =
            unsafe { xlib::XCreateGC(display, window_id, 0, &gc_values as *const _ as *mut _) };

        if clear_gc_raw_ptr.is_null() {
            // If this fails, xft_draw (SafeXftDraw) and pre_data will be dropped,
            // cleaning up their respective resources.
            return Err(anyhow!("XCreateGC failed for window ID {}", window_id));
        }
        let clear_gc = SafeGc::new(clear_gc_raw_ptr, display);
        debug!("Clear GC created: {:p}", clear_gc.raw());

        Ok(Self {
            xft_font: pre_data.xft_font, // Moved from pre_data
            xft_draw,                    // Newly created SafeXftDraw
            xft_ansi_colors: pre_data.xft_ansi_colors, // Moved from pre_data
            xft_color_cache_rgb: HashMap::new(),
            font_width_px: pre_data.font_width_px,
            font_height_px: pre_data.font_height_px,
            font_ascent_px: pre_data.font_ascent_px,
            clear_gc, // Newly created SafeGc
        })
    }

    /// Resolves a `crate::color::Color` to a reference to a concrete `xft::XftColor`.
    ///
    /// Handles named ANSI colors, indexed colors, and direct RGB colors.
    /// For RGB and indexed colors, it uses a cache (`xft_color_cache_rgb`)
    /// to avoid redundant allocations.
    ///
    /// # Returns
    /// * `Ok(xft::XftColor)`: An owned XftColor value.
    /// * `Err(anyhow::Error)`: If color allocation fails or `Color::Default` is passed.
    fn resolve_concrete_xft_color(
        &mut self,
        connection: &Connection,
        color: Color,
    ) -> Result<xft::XftColor> { // Returns an owned XftColor
        match color {
            Color::Default => {
                error!("Graphics::resolve_concrete_xft_color received Color::Default. This is a bug in the Renderer.");
                self.cached_rgb_to_xft_color(connection, 0, 0, 0)
                    // .map(|safe_color| safe_color.cloned_color()) // Already returns XftColor
                    .context("Fallback to black failed after Color::Default error")
            }
            Color::Named(named_color) => {
                Ok(self.xft_ansi_colors[named_color as u8 as usize].cloned_color())
            }
            Color::Indexed(idx) => {
                let rgb_equivalent = crate::color::convert_to_rgb_color(Color::Indexed(idx));
                if let Color::Rgb(r, g, b) = rgb_equivalent {
                    self.cached_rgb_to_xft_color(connection, r, g, b)
                        // .map(|safe_color| safe_color.cloned_color())
                } else {
                    error!(
                        "Failed to convert Indexed({}) to RGB. Defaulting to black.",
                        idx
                    );
                    self.cached_rgb_to_xft_color(connection, 0, 0, 0)
                        // .map(|safe_color| safe_color.cloned_color())
                        .context("Fallback to black failed after Indexed color conversion error")
                }
            }
            Color::Rgb(r, g, b) => self
                .cached_rgb_to_xft_color(connection, r, g, b),
                // .map(|safe_color| safe_color.cloned_color()),
        }
    }

    /// Retrieves or allocates a `SafeXftColor` for a given RGB value from the cache and returns a clone of its inner `xft::XftColor`.
    ///
    /// Uses an internal cache (`xft_color_cache_rgb`) to store and reuse `SafeXftColor`
    /// wrappers for previously seen RGB values, minimizing X server requests.
    ///
    /// # Returns
    /// * `Ok(xft::XftColor)`: An owned, cloned `xft::XftColor` value.
    /// * `Err(anyhow::Error)`: If `XftColorAllocValue` fails for a new color.
    fn cached_rgb_to_xft_color(
        &mut self,
        connection: &Connection,
        r_u8: u8,
        g_u8: u8,
        b_u8: u8,
    ) -> Result<xft::XftColor> { // Returns an owned XftColor
        // Check cache first.
        // Using entry API to avoid double hash lookup and handle borrowing correctly.
        if !self.xft_color_cache_rgb.contains_key(&(r_u8, g_u8, b_u8)) {
            // Color not in cache, allocate and insert it.
            let display = connection.display();
            let visual = connection.visual();
            let colormap = connection.colormap();

            let render_color = XRenderColor {
                red: ((r_u8 as u16) << 8) | (r_u8 as u16),
                green: ((g_u8 as u16) << 8) | (g_u8 as u16),
                blue: ((b_u8 as u16) << 8) | (b_u8 as u16),
                alpha: XRENDER_ALPHA_OPAQUE,
            };
            let mut new_xft_color_data: xft::XftColor = unsafe { mem::zeroed() };

            // SAFETY: FFI call.
            let success = unsafe {
                xft::XftColorAllocValue(
                    display,
                    visual,
                    colormap,
                    &render_color,
                    &mut new_xft_color_data,
                ) != 0
            };

            if success {
                let safe_color = SafeXftColor::new(
                    new_xft_color_data,
                    display,
                    visual,
                    colormap,
                    true, // Mark as allocated
                );
                self.xft_color_cache_rgb.insert((r_u8, g_u8, b_u8), safe_color);
            } else {
                return Err(anyhow!(
                    "XftColorAllocValue failed for RGB({},{},{})",
                    r_u8,
                    g_u8,
                    b_u8
                ));
            }
        }
        // Return a clone of the inner XftColor from the cached SafeXftColor.
        // .unwrap() is safe here because we've just inserted it if it wasn't there.
        Ok(self.xft_color_cache_rgb.get(&(r_u8, g_u8, b_u8)).unwrap().cloned_color())
    }

    /// Clears the entire drawable area (associated with `self.xft_draw.raw()`) with a specified background color.
    ///
    /// TODO: This method currently uses placeholder dimensions (800x600) for clearing.
    /// It needs to be updated to accept the current window width and height as arguments,
    /// which should be provided by `XDriver` (obtained from `Window::current_dimensions_pixels`).
    ///
    /// # Arguments
    /// * `connection`: A reference to the active X11 `Connection`.
    /// * `bg`: The `Color` to use for clearing the background. Assumed to be a concrete color.
    ///
    /// # Returns
    /// * `Ok(())` if successful, or an error if color resolution fails.
    pub(super) fn clear_all(&mut self, connection: &Connection, bg: Color) -> Result<()> {
        let xft_bg_color = self // This now returns an owned xft::XftColor
            .resolve_concrete_xft_color(connection, bg)
            .context("Failed to resolve background color for clear_all")?;

        // TODO: Replace placeholder dimensions with actual window dimensions.
        let placeholder_width = 800;
        let placeholder_height = 600;
        warn!(
            "Graphics::clear_all using placeholder dimensions {}x{}. Update required.",
            placeholder_width, placeholder_height
        );

        // SAFETY: Xft FFI call. `self.xft_draw.raw()` must be valid.
        unsafe {
            xft::XftDrawRect(
                self.xft_draw.raw(), // Use raw pointer from SafeXftDraw
                &xft_bg_color,       // Pass a reference to the owned XftColor
                0,
                0,
                placeholder_width,
                placeholder_height,
            );
        }
        trace!(
            "Window cleared (with placeholder dimensions) using color pixel: {}",
            xft_bg_color.pixel
        );
        Ok(())
    }

    /// Draws a run of text characters at specified cell coordinates with a given style.
    ///
    /// The background of the text run is filled first, then the text is drawn on top.
    /// Handles underline and strikethrough attributes from `style.flags`.
    ///
    /// # Arguments
    /// * `connection`: A reference to the active X11 `Connection`.
    /// * `coords`: `CellCoords` specifying the starting (column, row) for the text.
    /// * `text`: The string slice to draw.
    /// * `style`: `TextRunStyle` defining foreground/background colors and text attributes.
    ///
    /// # Returns
    /// * `Ok(())` if successful, or an error if color resolution or CString conversion fails.
    pub(super) fn draw_text_run(
        &mut self,
        connection: &Connection,
        coords: CellCoords,
        text: &str,
        style: TextRunStyle,
    ) -> Result<()> {
        if text.is_empty() {
            return Ok(()); // Nothing to draw.
        }

        // Calculate pixel coordinates for the text run's top-left corner.
        let x_pixel = (coords.x * self.font_width_px as usize) as c_int;
        let y_pixel = (coords.y * self.font_height_px as usize) as c_int;
        // Calculate width of the background rectangle for the text run in pixels.
        let run_pixel_width = text.chars().count() * self.font_width_px as usize;

        // Resolve foreground and background colors to owned XftColor values.
        let xft_fg = self
            .resolve_concrete_xft_color(connection, style.fg)
            .context("Failed to resolve foreground color for text run")?;
        let xft_bg = self
            .resolve_concrete_xft_color(connection, style.bg)
            .context("Failed to resolve background color for text run")?;

        // Draw the background rectangle.
        // SAFETY: Xft FFI call. `self.xft_draw.raw()` must be valid.
        unsafe {
            xft::XftDrawRect(
                self.xft_draw.raw(),
                &xft_bg, // Pass a reference to the owned XftColor
                x_pixel,
                y_pixel,
                run_pixel_width as u32,
                self.font_height_px,
            );
        }

        let c_text = CString::new(text).context("Failed to convert text to CString for Xft")?;
        let baseline_y_pixel = y_pixel + self.font_ascent_px as c_int;

        // Draw the text string.
        // SAFETY: Xft FFI call. Raw pointers from safe types must be valid.
        unsafe {
            xft::XftDrawStringUtf8(
                self.xft_draw.raw(),
                &xft_fg,              // Pass a reference to the owned XftColor
                self.xft_font.raw(),  // Use raw pointer from SafeXftFont
                x_pixel,
                baseline_y_pixel,
                c_text.as_ptr() as *const u8,
                c_text.as_bytes().len() as c_int,
            );
        }

        if style.flags.contains(AttrFlags::UNDERLINE) {
            let underline_y = y_pixel + self.font_height_px as c_int - 2;
            // SAFETY: Xft FFI call.
            unsafe {
                xft::XftDrawRect(
                    self.xft_draw.raw(),
                    &xft_fg, // Pass a reference
                    x_pixel,
                    underline_y,
                    run_pixel_width as u32,
                    1, /* thickness */
                );
            }
        }
        if style.flags.contains(AttrFlags::STRIKETHROUGH) {
            let strikethrough_y = y_pixel + (self.font_ascent_px / 2) as c_int;
            // SAFETY: Xft FFI call.
            unsafe {
                xft::XftDrawRect(
                    self.xft_draw.raw(),
                    &xft_fg, // Pass a reference
                    x_pixel,
                    strikethrough_y,
                    run_pixel_width as u32,
                    1, /* thickness */
                );
            }
        }
        Ok(())
    }

    /// Fills a rectangular area of cells with a specified color.
    ///
    /// # Arguments
    /// * `connection`: A reference to the active X11 `Connection`.
    /// * `rect`: `CellRect` defining the area to fill in cell coordinates.
    /// * `color`: The `Color` to fill the rectangle with. Assumed to be a concrete color.
    ///
    /// # Returns
    /// * `Ok(())` if successful, or an error if color resolution fails.
    pub(super) fn fill_rect(
        &mut self,
        connection: &Connection,
        rect: CellRect,
        color: Color,
    ) -> Result<()> {
        if rect.width == 0 || rect.height == 0 {
            return Ok(()); // Nothing to fill.
        }

        // Calculate pixel dimensions for the rectangle.
        let x_pixel = (rect.x * self.font_width_px as usize) as c_int;
        let y_pixel = (rect.y * self.font_height_px as usize) as c_int;
        let rect_pixel_width = (rect.width * self.font_width_px as usize) as u32;
        let rect_pixel_height = (rect.height * self.font_height_px as usize) as u32;

        // Resolve the concrete Color to an owned XftColor.
        let xft_fill_color = self
            .resolve_concrete_xft_color(connection, color)
            .context("Failed to resolve color for fill_rect")?;

        // SAFETY: Xft FFI call. `self.xft_draw.raw()` must be valid.
        unsafe {
            xft::XftDrawRect(
                self.xft_draw.raw(),
                &xft_fill_color, // Pass a reference to the owned XftColor
                x_pixel,
                y_pixel,
                rect_pixel_width,
                rect_pixel_height,
            );
        }
        Ok(())
    }

    /// Cleans up graphics resources.
    ///
    /// With RAII wrappers, this method's primary role is to explicitly clear
    /// collections like `Vec` and `HashMap` which own `SafeXftColor` instances.
    /// Doing so runs their `Drop` implementations earlier than `Graphics::drop`.
    /// The direct resource fields (`xft_font`, `xft_draw`, `clear_gc`) are
    /// managed by their respective `Safe` types' `Drop` implementations if this method
    /// doesn't cause them to be dropped (e.g. by nulling out their pointers, which
    /// the current Safe types do in their Drop impls).
    /// This method is idempotent.
    pub(super) fn cleanup(&mut self, _connection: &Connection) -> Result<()> {
        info!("Graphics::cleanup called. Explicitly clearing collections and main resources.");

        // Clearing these collections will trigger the Drop impl for each SafeXftColor.
        if !self.xft_ansi_colors.is_empty() {
            trace!("Clearing xft_ansi_colors Vec in cleanup.");
            self.xft_ansi_colors.clear(); // Drops each SafeXftColor
        }
        if !self.xft_color_cache_rgb.is_empty() {
            trace!("Clearing xft_color_cache_rgb HashMap in cleanup.");
            self.xft_color_cache_rgb.clear(); // Drops each SafeXftColor
        }

        // For the main resources, their Drop implementations will be called when Graphics
        // itself is dropped. If we want to force cleanup here and make Graphics::drop a no-op
        // for these, we would need to wrap them in Option and take them, or have specific
        // cleanup methods within the Safe types that nullify their internal pointers.
        // The current Safe types nullify their pointers in their Drop impls.
        // So, if Graphics::cleanup is called, then when Graphics::drop runs, the Drop impls
        // of SafeXftFont, etc., will find their pointers already null and do nothing more.
        // To make `cleanup` truly do the full cleanup and `drop` be a simpler log,
        // we can manually drop the fields here. This is not strictly necessary if we
        // are okay with drop doing the work, but can make behavior more explicit.
        // However, Rust's ownership typically means you don't manually drop like this
        // unless you are taking the resource out of an Option or using mem::take.
        // Let's assume the Drop impls of the Safe types are sufficient.
        // The `cleanup` method here mainly serves to clear collections early.

        // To ensure that if cleanup is called, the Drop of Graphics doesn't try to
        // re-log or imply resources are still there, we can modify the Drop message
        // or add flags. For now, the Safe types' Drop impls becoming no-ops after
        // their resources are freed (by nulling ptrs) is key.

        info!("Graphics cleanup of collections complete. Main resources will be cleaned by their own Drop impls when Graphics is dropped (or if already manually cleaned).");
        Ok(())
    }

    // --- Getter methods ---

    /// Returns the dimensions (width, height) of a single character cell in pixels.
    #[inline]
    pub(super) fn font_dimensions_pixels(&self) -> (u32, u32) {
        (self.font_width_px, self.font_height_px)
    }
}

/// Handles resource cleanup for `Graphics` instances.
///
/// The actual resource freeing is handled by the `Drop` implementations of the
/// `SafeXftFont`, `SafeXftDraw`, `SafeGc` fields, and the `SafeXftColor` elements
/// within the `Vec` and `HashMap`. This `Drop` implementation primarily serves for logging.
impl Drop for Graphics {
    fn drop(&mut self) {
        // The resources are managed by the RAII wrappers.
        // Their Drop implementations will be called automatically when `Graphics` is dropped.
        // This log message confirms that the Graphics object is being dropped.
        // The individual Safe types will log their own cleanup actions if trace logging is enabled.
        info!(
            "Graphics instance dropped. Font ptr: {:?}, Draw ptr: {:?}, GC valid: {}, ANSI colors len: {}, RGB cache len: {}. Cleanup relies on Drop impls of RAII wrappers.",
            self.xft_font.raw(),
            self.xft_draw.raw(),
            self.clear_gc.is_valid, // Use is_valid for SafeGc
            self.xft_ansi_colors.len(),
            self.xft_color_cache_rgb.len()
        );
    }
}
