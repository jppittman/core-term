use crate::color::Color;
use crate::glyph::AttrFlags;
use std::os::unix::io::RawFd;

/// Holds the current state of the platform, including display metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct PlatformState {
    pub event_fd: Option<RawFd>,
    pub font_cell_width_px: usize,
    pub font_cell_height_px: usize,
    pub scale_factor: f64,
    pub display_width_px: u16,
    pub display_height_px: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderCommand {
    ClearAll { bg: Color },
    DrawTextRun {
        x: usize,
        y: usize,
        text: String,
        fg: Color,
        bg: Color,
        flags: AttrFlags,
        is_selected: bool,
    },
    FillRect {
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        color: Color,
        is_selection_bg: bool,
    },
    SetCursorVisibility { visible: bool },
    SetWindowTitle { title: String },
    RingBell,
    PresentFrame,
}
