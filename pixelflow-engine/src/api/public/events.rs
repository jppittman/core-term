//! Public API events and state types.

use crate::input::{CursorIcon, KeySymbol, Modifiers, MouseButton};
use pixelflow_core::traits::Surface;
use pixelflow_render::Pixel;

// =============================================================================
// Engine → App Events (categorized by actor-scheduler lane)
// =============================================================================

/// Data lane events from engine to app (low priority, burst-limited).
#[derive(Debug, Clone)]
pub enum EngineEventData {
    /// Engine requests application to render a frame.
    RequestFrame {
        frame_id: u64,
        timestamp: std::time::Instant,
        target_timestamp: std::time::Instant,
        refresh_interval: std::time::Duration,
    },
}

/// Control lane events from engine to app (high priority, unlimited).
/// Time-critical events that must be processed immediately.
#[derive(Debug, Clone)]
pub enum EngineEventControl {
    /// Window was resized by the user or OS.
    Resize(u32, u32),
    /// Display scale factor changed (e.g., moved to different DPI monitor).
    ScaleChanged(f64),
    /// OS requested app close.
    CloseRequested,
}

/// Management lane events from engine to app (medium priority, unlimited).
/// User input and window events.
#[derive(Debug, Clone)]
pub enum EngineEventManagement {
    /// User pressed a key.
    KeyDown {
        key: KeySymbol,
        mods: Modifiers,
        text: Option<String>,
    },
    /// User moved/clicked mouse.
    MouseClick { x: u32, y: u32, button: MouseButton },
    /// Mouse move
    MouseMove { x: u32, y: u32, mods: Modifiers },
    /// Mouse release
    MouseRelease { x: u32, y: u32, button: MouseButton },
    /// Mouse scroll wheel.
    MouseScroll {
        x: u32,
        y: u32,
        dx: f32,
        dy: f32,
        mods: Modifiers,
    },
    /// Paste text.
    Paste(String),
    /// Focus gained.
    FocusGained,
    /// Focus lost.
    FocusLost,
    /// The application explicitly woke the loop (e.g. from PTY thread).
    Wake,
}

/// Unified event wrapper for the Application trait.
///
/// Pre-categorized by actor-scheduler lane so apps can easily route to the correct priority.
#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// Low priority event (burst-limited)
    Data(EngineEventData),
    /// High priority event (time-critical)
    Control(EngineEventControl),
    /// Medium priority event (user input)
    Management(EngineEventManagement),
}


// =============================================================================
// App → Engine Messages (categorized by actor-scheduler lane)
// =============================================================================

/// Data lane messages from app to engine (low priority, burst-limited).
///
/// Apps send rendered surfaces via the Data lane of the engine's ActorHandle.
/// This is the primary output from applications.
pub enum AppData<P: Pixel> {
    /// Rendered surface to present.
    RenderSurface {
        frame_id: u64,
        surface: Box<dyn Surface<P> + Send + Sync>,
        app_submit_time: std::time::Instant,
    },
}

/// Management lane messages from app to engine (medium priority, unlimited).
///
/// Apps send window management actions via the Management lane.
#[derive(Debug)]
pub enum AppManagement {
    /// Update the window title.
    SetTitle(String),
    /// Request a window resize.
    ResizeRequest(u32, u32),
    /// Change cursor icon.
    SetCursorIcon(CursorIcon),
    /// Copy text to clipboard.
    CopyToClipboard(String),
    /// Request paste from clipboard.
    RequestPaste,
}

// Note: Quit is sent via Control lane using engine's internal EngineControl::Quit variant
