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
use x11::{xft, xlib};
use x11::xrender::{XGlyphInfo, XRenderColor};
use libc::{c_char, c_int};


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
struct PreGraphicsData {
    xft_font: *mut xft::XftFont,
    font_width_px: u32,
    font_height_px: u32,
    font_ascent_px: u32,
    xft_ansi_colors: Vec<xft::XftColor>,
    default_bg_pixel_value: xlib::Atom,
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
    xft_font: *mut xft::XftFont,
    xft_draw: *mut xft::XftDraw,
    xft_ansi_colors: Vec<xft::XftColor>,
    xft_color_cache_rgb: HashMap<(u8, u8, u8), xft::XftColor>,
    font_width_px: u32,
    font_height_px: u32,
    font_ascent_px: u32,
    clear_gc: xlib::GC,
    default_bg_pixel_value: xlib::Atom,
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
    pub fn load_font_and_colors(connection: &Connection) -> Result<PreGraphicsData> {
        info!("Graphics: Loading font and pre-allocating colors.");
        let display = connection.display();
        let screen = connection.screen();

        // 1. Load Font & Metrics
        debug!("Loading font: {}", DEFAULT_FONT_NAME);
        let font_name_cstr = CString::new(DEFAULT_FONT_NAME)
            .context("Failed to create CString for font name")?;

        // SAFETY: Xlib/Xft FFI call. `display` and `screen` must be valid.
        let xft_font = unsafe { xft::XftFontOpenName(display, screen, font_name_cstr.as_ptr()) };
        if xft_font.is_null() {
            return Err(anyhow!(
                "XftFontOpenName failed for font: '{}'. Ensure font is installed and accessible.", DEFAULT_FONT_NAME
            ));
        }
        debug!("Font '{}' loaded successfully: {:p}", DEFAULT_FONT_NAME, xft_font);

        // SAFETY: Accessing fields of a valid `xft_font` pointer.
        let font_height_px = unsafe { ((*xft_font).ascent + (*xft_font).descent) as u32 };
        let font_ascent_px = unsafe { (*xft_font).ascent as u32 };

        let mut extents: XGlyphInfo = unsafe { mem::zeroed() };
        // Using "M" is a common way to estimate average character width.
        let sample_char_cstr = CString::new("M").expect("CString::new for 'M' should not fail.");
        // SAFETY: Xlib/Xft FFI call. `display` and `xft_font` must be valid.
        unsafe {
            xft::XftTextExtentsUtf8(
                display,
                xft_font,
                sample_char_cstr.as_ptr() as *const u8,
                sample_char_cstr.as_bytes().len() as c_int,
                &mut extents,
            );
        }
        let font_width_px = extents.xOff as u32; // xOff is the advance width.

        if font_width_px < MIN_FONT_WIDTH || font_height_px < MIN_FONT_HEIGHT {
            // Ensure font is closed if metrics are invalid.
            // SAFETY: Xlib/Xft FFI call. `display` and `xft_font` must be valid.
            unsafe { xft::XftFontClose(display, xft_font); }
            return Err(anyhow!(
                "Font dimensions (W:{}, H:{}) are below minimum requirements (W:{}, H:{}).",
                font_width_px, font_height_px, MIN_FONT_WIDTH, MIN_FONT_HEIGHT
            ));
        }
        info!(
            "Font metrics determined: Width={}, Height={}, Ascent={}",
            font_width_px, font_height_px, font_ascent_px
        );

        // 2. Initialize ANSI Colors
        let mut xft_ansi_colors = Vec::with_capacity(ANSI_COLOR_COUNT);
        // SAFETY: `set_len` is safe here as `XftColorAllocValue` will initialize each element.
        // Vec elements are `MaybeUninit<XftColor>` effectively until written by Xft.
        unsafe { xft_ansi_colors.set_len(ANSI_COLOR_COUNT); }

        for i in 0..ANSI_COLOR_COUNT {
            let named_color_enum = NamedColor::from_index(i as u8);
            let rgb_color = named_color_enum.to_rgb_color(); // Converts to `crate::color::Color::Rgb`
            let (r_u8, g_u8, b_u8) = match rgb_color {
                Color::Rgb(r, g, b) => (r, g, b),
                _ => {
                    // This case should not be reached if `to_rgb_color` is correct.
                    warn!("NamedColor::to_rgb_color for index {} did not return Color::Rgb. Defaulting to black.", i);
                    (0,0,0) // Default to black if conversion fails.
                }
            };
            // Convert 8-bit RGB to 16-bit for XRenderColor (0xRRGGBB -> 0xRRRRGGGGBBBB).
            let color_components = Rgb16Components {
                r: ((r_u8 as u16) << 8) | (r_u8 as u16),
                g: ((g_u8 as u16) << 8) | (g_u8 as u16),
                b: ((b_u8 as u16) << 8) | (b_u8 as u16),
            };

            let render_color = XRenderColor {
                red: color_components.r,
                green: color_components.g,
                blue: color_components.b,
                alpha: XRENDER_ALPHA_OPAQUE, // Fully opaque
            };

            // SAFETY: Xlib/Xft FFI call. `display`, `visual`, `colormap` from connection must be valid.
            // `xft_ansi_colors[i]` provides a mutable pointer to an element in the Vec.
            if unsafe {
                xft::XftColorAllocValue(
                    display,
                    connection.visual(),
                    connection.colormap(),
                    &render_color, // const pointer
                    &mut xft_ansi_colors[i], // mutable pointer
                )
            } == 0 { // XftColorAllocValue returns 0 on failure.
                // Cleanup partially allocated resources before returning error.
                // SAFETY: Xlib/Xft FFI call.
                unsafe { xft::XftFontClose(display, xft_font); }
                for j in 0..i { // Free already allocated colors.
                     // SAFETY: Xlib/Xft FFI call.
                     unsafe { xft::XftColorFree(display, connection.visual(), connection.colormap(), &mut xft_ansi_colors[j]); }
                }
                return Err(anyhow!("XftColorAllocValue failed for ANSI color index {}", i));
            }
        }
        debug!("Preallocated ANSI Xft colors initialized.");

        // Determine default background pixel value from the pre-allocated black.
        let default_bg_pixel_value = xft_ansi_colors[NamedColor::Black as usize].pixel;
        info!("Default background pixel value (from Black ANSI color): {}", default_bg_pixel_value);

        Ok(PreGraphicsData {
            xft_font,
            font_width_px,
            font_height_px,
            font_ascent_px,
            xft_ansi_colors,
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
    pub fn new(
        connection: &Connection,
        window_id: xlib::Window,
        pre_data: PreGraphicsData,
    ) -> Result<Self> {
        info!("Graphics: Finalizing initialization with window ID: {}", window_id);
        let display = connection.display();

        // 1. Create XftDraw object for the window.
        // SAFETY: Xlib/Xft FFI call. `display`, `window_id`, `visual`, `colormap` must be valid.
        let xft_draw = unsafe {
            xft::XftDrawCreate(
                display,
                window_id, // Draw onto this specific window
                connection.visual(),
                connection.colormap(),
            )
        };
        if xft_draw.is_null() {
            // Cleanup resources from pre_data if XftDraw creation fails.
            // SAFETY: Xlib/Xft FFI calls.
            unsafe {
                xft::XftFontClose(display, pre_data.xft_font);
                for mut color in pre_data.xft_ansi_colors { // `Vec` owns `XftColor`s now.
                    xft::XftColorFree(display, connection.visual(), connection.colormap(), &mut color);
                }
            }
            return Err(anyhow!("Failed to create XftDraw object for window ID {}", window_id));
        }
        debug!("XftDraw object created: {:p}", xft_draw);

        // 2. Create Graphics Context (GC) for simple operations like clearing background.
        // Note: XftDraw is generally preferred for text and colored rects with XRender.
        // A simple GC might be used for other operations if ever needed.
        let gc_values: xlib::XGCValues = unsafe { mem::zeroed() }; // Initialize with defaults.
        // SAFETY: Xlib FFI call. `display` and `window_id` must be valid.
        let clear_gc = unsafe {
            xlib::XCreateGC(display, window_id, 0, &gc_values as *const _ as *mut _)
        };
        if clear_gc.is_null() {
            // Cleanup previously created XftDraw and pre_data resources.
            // SAFETY: Xlib/Xft FFI calls.
            unsafe {
                xft::XftDrawDestroy(xft_draw);
                xft::XftFontClose(display, pre_data.xft_font);
                for mut color in pre_data.xft_ansi_colors {
                    xft::XftColorFree(display, connection.visual(), connection.colormap(), &mut color);
                }
            }
            return Err(anyhow!("XCreateGC failed for window ID {}", window_id));
        }
        debug!("Clear GC created: {:p}", clear_gc);

        Ok(Self {
            xft_font: pre_data.xft_font,
            xft_draw,
            xft_ansi_colors: pre_data.xft_ansi_colors,
            xft_color_cache_rgb: HashMap::new(), // Initialize cache for dynamic colors
            font_width_px: pre_data.font_width_px,
            font_height_px: pre_data.font_height_px,
            font_ascent_px: pre_data.font_ascent_px,
            clear_gc,
            default_bg_pixel_value: pre_data.default_bg_pixel_value,
        })
    }

    /// Resolves a `crate::color::Color` to a concrete `xft::XftColor`.
    ///
    /// Handles named ANSI colors (from pre-allocated cache), indexed colors (approximated to RGB),
    /// and direct RGB colors. For RGB and indexed colors, it uses a cache (`xft_color_cache_rgb`)
    /// to avoid redundant allocations of `XftColor` structures.
    ///
    /// # Arguments
    /// * `connection`: A reference to the active X11 `Connection` (for display, visual, colormap access).
    /// * `color`: The `crate::color::Color` to resolve.
    ///
    /// # Returns
    /// * `Ok(xft::XftColor)`: The resolved Xft color.
    /// * `Err(anyhow::Error)`: If color allocation fails for RGB/indexed colors, or if `Color::Default`
    ///   is passed (which is considered a bug in the calling Renderer).
    fn resolve_concrete_xft_color(
        &mut self,
        connection: &Connection,
        color: Color,
    ) -> Result<xft::XftColor> {
        match color {
            Color::Default => {
                // The Renderer component is responsible for resolving default fg/bg colors
                // before they reach the driver. Receiving Color::Default here indicates a logic error upstream.
                error!("Graphics::resolve_concrete_xft_color received Color::Default. This is a bug in the Renderer.");
                // Fallback to black for safety, but this should be investigated.
                self.cached_rgb_to_xft_color(connection, 0, 0, 0)
                    .context("Fallback to black failed after Color::Default error")
            }
            Color::Named(named_color) => {
                // Standard ANSI colors are pre-allocated.
                Ok(self.xft_ansi_colors[named_color as u8 as usize])
            }
            Color::Indexed(idx) => {
                // Convert indexed color to its RGB equivalent first.
                let rgb_equivalent = crate::color::convert_to_rgb_color(Color::Indexed(idx));
                if let Color::Rgb(r, g, b) = rgb_equivalent {
                    self.cached_rgb_to_xft_color(connection, r, g, b)
                } else {
                    // Should not happen if convert_to_rgb_color is correct.
                    error!("Failed to convert Indexed({}) to RGB. Defaulting to black.", idx);
                    self.cached_rgb_to_xft_color(connection, 0, 0, 0)
                        .context("Fallback to black failed after Indexed color conversion error")
                }
            }
            Color::Rgb(r, g, b) => {
                self.cached_rgb_to_xft_color(connection, r, g, b)
            }
        }
    }

    /// Retrieves or allocates an `XftColor` for a given RGB value.
    ///
    /// Uses an internal cache (`xft_color_cache_rgb`) to store and reuse `XftColor`
    /// structures for previously seen RGB values, minimizing X server requests.
    ///
    /// # Arguments
    /// * `connection`: A reference to the active X11 `Connection`.
    /// * `r_u8`, `g_u8`, `b_u8`: The 8-bit red, green, and blue components of the color.
    ///
    /// # Returns
    /// * `Ok(xft::XftColor)`: The cached or newly allocated Xft color.
    /// * `Err(anyhow::Error)`: If `XftColorAllocValue` fails for a new color.
    fn cached_rgb_to_xft_color(
        &mut self,
        connection: &Connection,
        r_u8: u8,
        g_u8: u8,
        b_u8: u8,
    ) -> Result<xft::XftColor> {
        // Check cache first.
        if let Some(cached_color) = self.xft_color_cache_rgb.get(&(r_u8, g_u8, b_u8)) {
            return Ok(*cached_color); // Return a copy of the cached XftColor.
        }

        // Convert 8-bit RGB to 16-bit per component for XRenderColor.
        let color_components = Rgb16Components {
            r: ((r_u8 as u16) << 8) | (r_u8 as u16), // e.g., 0xAB -> 0xABAB
            g: ((g_u8 as u16) << 8) | (g_u8 as u16),
            b: ((b_u8 as u16) << 8) | (b_u8 as u16),
        };
        let render_color = XRenderColor {
            red: color_components.r,
            green: color_components.g,
            blue: color_components.b,
            alpha: XRENDER_ALPHA_OPAQUE, // Fully opaque
        };
        let mut new_xft_color: xft::XftColor = unsafe { mem::zeroed() };

        // SAFETY: Xlib/Xft FFI call. Connection members must be valid.
        if unsafe {
            xft::XftColorAllocValue(
                connection.display(),
                connection.visual(),
                connection.colormap(),
                &render_color,      // const pointer
                &mut new_xft_color, // mutable pointer
            )
        } == 0 { // XftColorAllocValue returns 0 on failure.
            Err(anyhow!("XftColorAllocValue failed for RGB({},{},{})", r_u8, g_u8, b_u8))
        } else {
            // Store the newly allocated color in the cache.
            self.xft_color_cache_rgb.insert((r_u8, g_u8, b_u8), new_xft_color);
            Ok(new_xft_color)
        }
    }

    /// Clears the entire drawable area (associated with `self.xft_draw`) with a specified background color.
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
    pub fn clear_all(&mut self, connection: &Connection, bg: Color) -> Result<()> {
        let xft_bg_color = self.resolve_concrete_xft_color(connection, bg)
            .context("Failed to resolve background color for clear_all")?;

        // TODO: Replace placeholder dimensions with actual window dimensions passed as arguments.
        // These dimensions should come from `Window::current_dimensions_pixels()` via `XDriver`.
        let placeholder_width = 800;
        let placeholder_height = 600;
        warn!(
            "Graphics::clear_all using placeholder dimensions {}x{}. Update required.",
            placeholder_width, placeholder_height
        );

        // SAFETY: Xft FFI call. `self.xft_draw` must be valid.
        unsafe {
             xft::XftDrawRect(
                self.xft_draw,
                &xft_bg_color,
                0, 0, // x, y
                placeholder_width, placeholder_height // width, height
            );
        }
        trace!("Window cleared (with placeholder dimensions) using color pixel: {}", xft_bg_color.pixel);
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
    pub fn draw_text_run(
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

        // Resolve foreground and background colors to XftColor.
        let xft_fg = self.resolve_concrete_xft_color(connection, style.fg)
            .context("Failed to resolve foreground color for text run")?;
        let xft_bg = self.resolve_concrete_xft_color(connection, style.bg)
            .context("Failed to resolve background color for text run")?;

        // Draw the background rectangle for the text run.
        // SAFETY: Xft FFI call. `self.xft_draw` must be valid.
        unsafe {
            xft::XftDrawRect(
                self.xft_draw,
                &xft_bg, // Use the resolved background color
                x_pixel,
                y_pixel,
                run_pixel_width as u32, // Width of the text run
                self.font_height_px,    // Height of a single line/cell
            );
        }

        // Convert the Rust string to a C-compatible string for Xft.
        let c_text = CString::new(text).context("Failed to convert text to CString for Xft")?;
        // Calculate the baseline Y coordinate for XftDrawStringUtf8.
        let baseline_y_pixel = y_pixel + self.font_ascent_px as c_int;

        // Draw the text string.
        // SAFETY: Xft FFI call. `self.xft_draw` and `self.xft_font` must be valid.
        unsafe {
            xft::XftDrawStringUtf8(
                self.xft_draw,
                &xft_fg,          // Use the resolved foreground color
                self.xft_font,    // The loaded Xft font
                x_pixel,          // X position in pixels
                baseline_y_pixel, // Y position of the baseline in pixels
                c_text.as_ptr() as *const u8, // Text as C string
                c_text.as_bytes().len() as c_int, // Length of the text in bytes
            );
        }

        // Handle underline and strikethrough attributes if present.
        if style.flags.contains(AttrFlags::UNDERLINE) {
            // Simple underline: 1 pixel high, positioned near the bottom of the cell.
            let underline_y = y_pixel + self.font_height_px as c_int - 2; // Adjust position as needed.
            // SAFETY: Xft FFI call.
            unsafe {
                xft::XftDrawRect(self.xft_draw, &xft_fg, x_pixel, underline_y, run_pixel_width as u32, 1 /* thickness */);
            }
        }
        if style.flags.contains(AttrFlags::STRIKETHROUGH) {
            // Simple strikethrough: 1 pixel high, roughly in the middle of the text ascent.
            let strikethrough_y = y_pixel + (self.font_ascent_px / 2) as c_int; // Adjust position as needed.
            // SAFETY: Xft FFI call.
            unsafe {
                xft::XftDrawRect(self.xft_draw, &xft_fg, x_pixel, strikethrough_y, run_pixel_width as u32, 1 /* thickness */);
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
    pub fn fill_rect(
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

        // Resolve the concrete Color to an XftColor.
        let xft_fill_color = self.resolve_concrete_xft_color(connection, color)
            .context("Failed to resolve color for fill_rect")?;

        // SAFETY: Xft FFI call. `self.xft_draw` must be valid.
        unsafe {
            xft::XftDrawRect(
                self.xft_draw,
                &xft_fill_color,
                x_pixel, y_pixel,
                rect_pixel_width, rect_pixel_height,
            );
        }
        Ok(())
    }

    /// Cleans up all X11 graphics resources managed by this instance.
    ///
    /// This includes destroying the `XftDraw` object, closing the `XftFont`,
    /// freeing all allocated `XftColor`s (both pre-allocated ANSI colors and
    /// cached RGB colors), and freeing the Graphics Context (GC).
    /// This method is idempotent.
    ///
    /// # Arguments
    /// * `connection`: A reference to the active X11 `Connection` (needed for display, visual, colormap).
    ///
    /// # Returns
    /// * `Ok(())` always. Errors during freeing of X resources are logged but not propagated.
    pub fn cleanup(&mut self, connection: &Connection) -> Result<()> {
        info!("Cleaning up Graphics resources...");
        let display = connection.display();
        // SAFETY: Xlib/Xft FFI calls. Ensure connection members are valid.
        // Operations are checked for null pointers to ensure idempotency.
        unsafe {
            if !self.xft_draw.is_null() {
                trace!("Destroying XftDraw object: {:p}", self.xft_draw);
                xft::XftDrawDestroy(self.xft_draw);
                self.xft_draw = ptr::null_mut();
            }
            if !self.xft_font.is_null() {
                trace!("Closing XftFont: {:p}", self.xft_font);
                xft::XftFontClose(display, self.xft_font);
                self.xft_font = ptr::null_mut();
            }

            if !self.xft_ansi_colors.is_empty() {
                trace!("Freeing {} preallocated ANSI XftColors.", self.xft_ansi_colors.len());
                for color_ptr in self.xft_ansi_colors.iter_mut() {
                    // Check if pixel is non-zero or color components are non-zero before freeing,
                    // as XftColorFree might behave unexpectedly with entirely zeroed XftColor structs
                    // that were never successfully allocated.
                    if color_ptr.pixel != 0 || color_ptr.color.red != 0 || color_ptr.color.green != 0 || color_ptr.color.blue != 0 {
                        xft::XftColorFree(display, connection.visual(), connection.colormap(), color_ptr);
                    }
                }
                self.xft_ansi_colors.clear();
            }

            if !self.xft_color_cache_rgb.is_empty() {
                trace!("Freeing {} cached RGB XftColors.", self.xft_color_cache_rgb.len());
                for (_, mut cached_color) in self.xft_color_cache_rgb.drain() {
                    xft::XftColorFree(display, connection.visual(), connection.colormap(), &mut cached_color);
                }
                // HashMap::drain clears the map.
            }

            if !self.clear_gc.is_null() {
                trace!("Freeing clear GC: {:p}", self.clear_gc);
                xlib::XFreeGC(display, self.clear_gc);
                self.clear_gc = ptr::null_mut();
            }
        }
        info!("Graphics cleanup complete.");
        Ok(())
    }

    // --- Getter methods ---

    /// Returns the dimensions (width, height) of a single character cell in pixels.
    #[inline]
    pub fn font_dimensions_pixels(&self) -> (u32, u32) {
        (self.font_width_px, self.font_height_px)
    }

    /// Returns the pixel value used for the initial background of the window.
    /// This value is typically derived from the pre-allocated black ANSI color.
    #[inline]
    pub fn initial_background_pixel_value(&self) -> xlib::Atom {
        self.default_bg_pixel_value
    }
}

/// Handles resource cleanup for `Graphics` instances.
///
/// This `Drop` implementation serves as a safeguard. It logs an error if critical
/// X resources (font, draw object, GC, colors) appear unreleased, indicating that
/// `cleanup()` was not explicitly called. **It cannot safely call Xlib functions**
/// for cleanup because it lacks access to the `Connection`.
impl Drop for Graphics {
    fn drop(&mut self) {
        // Check if resources seem to be unreleased.
        if !self.xft_font.is_null()
            || !self.xft_draw.is_null()
            || !self.clear_gc.is_null()
            || !self.xft_ansi_colors.is_empty()
            || !self.xft_color_cache_rgb.is_empty() {
            error!(
                "Graphics dropped without explicit cleanup called by XDriver. X server resources may leak. Font: {:?}, Draw: {:?}, GC: {:?}, ANSI colors: {}, RGB cache: {}",
                self.xft_font, self.xft_draw, self.clear_gc, self.xft_ansi_colors.len(), self.xft_color_cache_rgb.len()
            );
        } else {
            info!("Graphics dropped (already cleaned up).");
        }
    }
}
