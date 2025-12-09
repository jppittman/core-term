//! Unified message types for pixelflow-engine.
//!
//! This module contains ALL internal message types, consolidating what was previously
//! scattered across `api/private/display.rs` and `api/private/channels.rs`.
//!
//! **Goal:** One definition per event type. No DisplayEvent::Key + EngineEventManagement::KeyDown duplication.

use crate::input::{KeySymbol, Modifiers};
use crate::vsync_actor::VsyncActor;
use pixelflow_core::Pixel;
use pixelflow_render::Frame;
use actor_scheduler::{ActorHandle, WakeHandler, WeakActorHandle, Message};
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use anyhow::Result;

// ============================================================================
// Window Identifier
// ============================================================================

/// Opaque window identifier for multi-window support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WindowId(pub u32);

impl WindowId {
    /// The default/primary window ID.
    pub const PRIMARY: WindowId = WindowId(0);
}

// ============================================================================
// Engine Handle Trait (Driver → Engine communication)
// ============================================================================

/// Trait for sending display events to the engine.
///
/// This is the interface the display driver uses to communicate with the engine.
/// The actual `ActorHandle` is "stronger" (can send Control/Management messages too),
/// but it satisfies this weaker contract that the driver needs.
///
/// Benefits:
/// - Driver only depends on what it needs (send DisplayEvent)
/// - Tests can provide mock implementations
/// - Enables passing WeakActorHandle to avoid reference cycles
pub trait EngineHandle: Send + Sync {
    /// The pixel format the engine expects.
    type Pixel: Pixel;

    /// Send a display event to the engine.
    fn send(&self, event: DisplayEvent) -> Result<()>;
}

/// ActorHandle implements EngineHandle natively.
impl<P: Pixel> EngineHandle for ActorHandle<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement> {
    type Pixel = P;

    fn send(&self, event: DisplayEvent) -> Result<()> {
        ActorHandle::send(self, Message::Data(EngineData::FromDriver(event)))
            .map_err(|_| anyhow::anyhow!("Failed to send event to engine"))
    }
}

/// WeakActorHandle implements EngineHandle (upgrades to strong, then sends).
impl<P: Pixel> EngineHandle for WeakActorHandle<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement> {
    type Pixel = P;

    fn send(&self, event: DisplayEvent) -> Result<()> {
        WeakActorHandle::send(self, Message::Data(EngineData::FromDriver(event)))
            .map_err(|_| anyhow::anyhow!("Engine disconnected"))
    }
}

// ============================================================================
// Platform Events (From Driver)
// ============================================================================

/// Platform-agnostic display events sent from driver to engine.
///
/// NOTE: This will be unified with app-facing events in Phase 2 of the refactor.
/// For now, keeping the existing DisplayEvent structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DisplayEvent {
    // ========================================================================
    // Lifecycle
    // ========================================================================
    /// Window was successfully created.
    WindowCreated {
        id: WindowId,
        width_px: u32,
        height_px: u32,
        scale: f64,
    },

    /// Window was destroyed.
    WindowDestroyed { id: WindowId },

    /// User requested window close (e.g., clicked X button).
    CloseRequested { id: WindowId },

    // ========================================================================
    // Resize / Scale
    // ========================================================================
    /// Window/framebuffer was resized (physical pixels).
    Resized {
        id: WindowId,
        width_px: u32,
        height_px: u32,
    },

    /// Display scale factor changed (e.g., moved to different DPI monitor).
    ScaleChanged { id: WindowId, scale: f64 },

    // ========================================================================
    // Input
    // ========================================================================
    /// Key press/release event.
    Key {
        id: WindowId,
        symbol: KeySymbol,
        modifiers: Modifiers,
        text: Option<String>,
    },

    // ========================================================================
    // Mouse
    // ========================================================================
    /// Mouse button press.
    MouseButtonPress {
        id: WindowId,
        button: u8,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },

    /// Mouse button release.
    MouseButtonRelease {
        id: WindowId,
        button: u8,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },

    /// Mouse movement.
    MouseMove {
        id: WindowId,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },

    /// Mouse scroll wheel.
    MouseScroll {
        id: WindowId,
        dx: f32,
        dy: f32,
        x: i32,
        y: i32,
        modifiers: Modifiers,
    },

    // ========================================================================
    // Focus
    // ========================================================================
    /// Window gained focus.
    FocusGained { id: WindowId },

    /// Window lost focus.
    FocusLost { id: WindowId },

    // ========================================================================
    // Clipboard (global, no window ID)
    // ========================================================================
    /// Paste data from clipboard.
    PasteData { text: String },

    /// Clipboard data requested by another application (X11 SelectionRequest).
    ClipboardDataRequested,
}

// ============================================================================
// Engine Data Lane (Incoming Events)
// ============================================================================

/// Data lane messages (low priority, burst-limited with backpressure).
/// Wraps messages from driver, app, and vsync.
pub enum EngineData<P: Pixel> {
    /// Platform event from driver (window, input, etc.)
    FromDriver(DisplayEvent),
    /// Rendered frame from app
    FromApp(crate::api::public::AppData<P>),
    /// VSync timing signal from vsync actor
    FromVSync(crate::vsync_actor::VsyncMessage),
}

impl<P: Pixel> From<DisplayEvent> for EngineData<P> {
    fn from(evt: DisplayEvent) -> Self {
        EngineData::FromDriver(evt)
    }
}

impl<P: Pixel> From<crate::api::public::AppData<P>> for EngineData<P> {
    fn from(data: crate::api::public::AppData<P>) -> Self {
        EngineData::FromApp(data)
    }
}

impl<P: Pixel> From<crate::vsync_actor::VsyncMessage> for EngineData<P> {
    fn from(msg: crate::vsync_actor::VsyncMessage) -> Self {
        EngineData::FromVSync(msg)
    }
}

// ============================================================================
// Engine Control Lane (Time-Critical Operations)
// ============================================================================

/// Control lane messages (high priority, unlimited drain).
/// Sent from driver to engine for time-critical operations, and from apps for critical actions.
#[derive(Debug)]
pub enum EngineControl<P: Pixel> {
    /// Frame returned from driver after presentation
    PresentComplete(Frame<P>),

    /// Driver operation completed (SetTitle, etc.)
    DriverAck,

    /// Update refresh rate (from driver, for VRR or monitor changes)
    UpdateRefreshRate(f64),

    /// Quit request from application (high priority)
    Quit,

    /// Internal: Set engine handle (sent once after spawn for VSync creation)
    SetEngineHandle(WeakActorHandle<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement>),

    /// Internal: Register application handle (sent once during platform.run())
    RegisterApp(WeakActorHandle<crate::api::public::EngineEvent, (), crate::api::public::AppManagement>),
}

// Auto-convert DisplayEvent to Message (via EngineData) for ergonomic sending
impl<P: Pixel> From<DisplayEvent> for actor_scheduler::Message<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement> {
    fn from(evt: DisplayEvent) -> Self {
        actor_scheduler::Message::Data(EngineData::FromDriver(evt))
    }
}

// ============================================================================
// Driver Commands (Engine → Driver)
// ============================================================================

/// Commands sent TO the driver (from engine).
#[derive(Debug)]
pub enum DriverCommand<P: Pixel> {
    // ========================================================================
    // Lifecycle
    // ========================================================================
    /// Create a new window with the given ID and dimensions.
    CreateWindow {
        id: WindowId,
        width: u32,
        height: u32,
        title: String,
    },

    /// Destroy a window.
    DestroyWindow { id: WindowId },

    // ========================================================================
    // Rendering
    // ========================================================================
    /// Present a frame to a window.
    /// The Frame contains typed pixel data matching the driver's Pixel type.
    Present {
        id: WindowId,
        frame_id: u64,
        frame: Frame<P>,
        engine_submit_time: Instant,
    },

    // ========================================================================
    // Window properties
    // ========================================================================
    /// Set window title.
    SetTitle { id: WindowId, title: String },

    /// Set window size.
    SetSize {
        id: WindowId,
        width: u32,
        height: u32,
    },

    // ========================================================================
    // Clipboard
    // ========================================================================
    /// Copy text to clipboard.
    CopyToClipboard(String),

    /// Request paste from clipboard.
    RequestPaste,

    // ========================================================================
    // Audio
    // ========================================================================
    /// Ring the terminal bell.
    Bell,

    // ========================================================================
    // Control
    // ========================================================================
    /// Set the engine handle (injected after spawn to avoid circular reference).
    /// Driver stores this handle to send display events back to the engine.
    SetEngineHandle(WeakActorHandle<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement>),

    /// Shutdown the driver.
    Shutdown,
}

// ============================================================================
// ActorScheduler-based Engine Channels
// ============================================================================

/// Display event burst limit (max events processed per wake cycle)
pub const DISPLAY_EVENT_BURST_LIMIT: usize = 10;

/// Display event buffer size (backpressure threshold)
pub const DISPLAY_EVENT_BUFFER_SIZE: usize = 64;

/// Type alias for engine actor handle (sends messages TO engine)
pub type EngineActorHandle<P> = ActorHandle<EngineData<P>, EngineControl<P>, crate::api::public::AppManagement>;
