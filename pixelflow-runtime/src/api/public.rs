use crate::input::{KeySymbol, Modifiers, MouseButton};
// use pixelflow_render::Frame;

/// Window ID wrapper for identifying windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId(pub u64);

impl WindowId {
    /// Primary window ID (for single-window applications).
    pub const PRIMARY: Self = Self(0);
}

/// Events sent from the Engine to the Application.
///
/// The Engine emits events to communicate state changes and requests to the application.
/// Events are organized by priority: **Control** (most critical), **Management** (medium),
/// **Data** (high-frequency, lowest priority).
///
/// # Event Contract
///
/// When the Engine sends an event, it guarantees:
/// - **Timeliness**: Events are delivered as soon as feasible after the triggering action
/// - **Accuracy**: Event data accurately reflects the OS event or engine state
/// - **Ordering**: Events within a priority lane maintain causal order
///
/// The Application should handle events promptly to keep the rendering loop responsive.
#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// Control events (critical state changes, highest priority).
    ///
    /// These indicate important changes to the window or application state
    /// that may require immediate response.
    Control(EngineEventControl),

    /// Management events (input and user interactions, medium priority).
    ///
    /// These carry information about user input (keyboard, mouse) and clipboard.
    Management(EngineEventManagement),

    /// Data events (frame requests, lowest priority).
    ///
    /// These signal that it's time to render a frame.
    Data(EngineEventData),
}

/// Control events from the Engine (window state changes).
#[derive(Debug, Clone)]
pub enum EngineEventControl {
    /// Window was created by the driver.
    ///
    /// # Contract
    ///
    /// **Engine**: Relays WindowCreated event from driver to app.
    ///
    /// **Application**: Window is now ready for rendering. Should start VSync and begin sending frames.
    ///
    /// # Arguments
    ///
    /// - `id`: Window identifier for future references
    /// - `width_px`: Width in physical pixels
    /// - `height_px`: Height in physical pixels
    /// - `scale`: DPI scale factor
    WindowCreated {
        id: WindowId,
        width_px: u32,
        height_px: u32,
        scale: f64,
    },

    /// Window has been resized.
    ///
    /// # Contract
    ///
    /// **Engine**: Relays resize event from driver to app.
    ///
    /// **Application**: Should update its render target size and send new frames.
    /// The application may receive multiple `Resize` events before rendering a frame.
    ///
    /// # Arguments
    ///
    /// - `id`: Window identifier
    /// - `width_px`: Width in physical pixels
    /// - `height_px`: Height in physical pixels
    Resized {
        id: WindowId,
        width_px: u32,
        height_px: u32,
    },

    /// User requested to close the window.
    ///
    /// # Contract
    ///
    /// **Engine**: The user clicked the close button or pressed Alt+F4.
    ///
    /// **Application**: Should clean up and shut down gracefully. Not receiving
    /// this event doesn't mean the window is still open—the OS may force-close it.
    ///
    /// # Note
    ///
    /// This is a request, not a command. The application can ignore it (for unsaved
    /// changes confirmation), but the OS may force-close the window anyway.
    CloseRequested,

    /// DPI scale factor changed.
    ///
    /// # Contract
    ///
    /// **Engine**: Relays scale change event from driver to app.
    ///
    /// **Application**: May need to rerender at new resolution or adjust font sizes.
    /// The scale factor affects how logical pixels map to physical pixels.
    ///
    /// # Arguments
    ///
    /// - `id`: Window identifier
    /// - `scale`: Scale factor (e.g., 1.0 = 96 DPI, 2.0 = 192 DPI on high-DPI displays)
    ScaleChanged { id: WindowId, scale: f64 },
}

/// Management events from the Engine (input and interactions).
#[derive(Debug, Clone)]
pub enum EngineEventManagement {
    /// A key was pressed.
    ///
    /// # Contract
    ///
    /// **Engine**: User pressed a keyboard key; Engine provides symbol and modifiers.
    ///
    /// **Application**: Should process the keystroke and update state.
    /// If text input is needed, the `text` field contains the character(s) if available.
    ///
    /// # Arguments
    ///
    /// - `key`: Key symbol (arrow, letter, function key, etc.)
    /// - `mods`: Modifier keys (Shift, Ctrl, Alt, etc.)
    /// - `text`: Composed text character (Some for printable characters, None for control keys)
    KeyDown {
        key: KeySymbol,
        mods: Modifiers,
        text: Option<String>,
    },

    /// Mouse button pressed.
    ///
    /// # Contract
    ///
    /// **Engine**: User clicked a mouse button; coordinates are in logical pixels.
    ///
    /// **Application**: Should process the click (e.g., select text, open menu).
    ///
    /// # Arguments
    ///
    /// - `x`, `y`: Click position in logical pixels
    /// - `button`: Which button was clicked (left=0, right=1, middle=2, etc.)
    MouseClick { x: u32, y: u32, button: MouseButton },

    /// Mouse button released.
    ///
    /// # Contract
    ///
    /// **Engine**: User released a mouse button; coordinates match button location.
    ///
    /// **Application**: Should complete any click action started by the corresponding press.
    ///
    /// # Arguments
    ///
    /// - `x`, `y`: Release position in logical pixels
    /// - `button`: Which button was released
    MouseRelease { x: u32, y: u32, button: MouseButton },

    /// Mouse moved.
    ///
    /// # Contract
    ///
    /// **Engine**: Mouse pointer moved to a new position.
    ///
    /// **Application**: May update cursor icon, highlight selections, or track position.
    /// High-frequency event; application should handle efficiently.
    ///
    /// # Arguments
    ///
    /// - `x`, `y`: New position in logical pixels
    /// - `mods`: Current modifier key state
    MouseMove { x: u32, y: u32, mods: Modifiers },

    /// Mouse wheel or trackpad scrolled.
    ///
    /// # Contract
    ///
    /// **Engine**: User scrolled; delta is in logical units.
    ///
    /// **Application**: Should scroll content in the specified direction.
    /// Sign convention: Positive = scroll down/right, Negative = scroll up/left.
    ///
    /// # Arguments
    ///
    /// - `x`, `y`: Cursor position at time of scroll (logical pixels)
    /// - `dx`, `dy`: Scroll delta (touchpad may report fine-grained values)
    /// - `mods`: Modifier key state (Shift, Ctrl for alternate scroll behavior)
    MouseScroll {
        x: u32,
        y: u32,
        dx: f32,
        dy: f32,
        mods: Modifiers,
    },

    /// Window gained focus.
    ///
    /// # Contract
    ///
    /// **Engine**: Window is now the active (foreground) window.
    ///
    /// **Application**: May resume animations, activate input handlers, etc.
    FocusGained,

    /// Window lost focus.
    ///
    /// # Contract
    ///
    /// **Engine**: Window is no longer the active window.
    ///
    /// **Application**: May pause animations, suspend input processing, etc.
    FocusLost,

    /// Text pasted from clipboard.
    ///
    /// # Contract
    ///
    /// **Engine**: User pasted; text comes from the system clipboard.
    ///
    /// **Application**: Should insert the pasted text at the cursor position.
    ///
    /// # Arguments
    ///
    /// - UTF-8 text from the clipboard
    Paste(String),
}

/// Data events from the Engine (frame synchronization).
///
/// These are high-frequency events that drive the render loop.
#[derive(Debug, Clone)]
pub enum EngineEventData {
    /// Request a new frame.
    ///
    /// # Contract
    ///
    /// **Engine**: It's time to render a new frame; provides timing information.
    ///
    /// **Application**: Should render and send a frame via `AppData::RenderSurface`.
    ///
    /// # Arguments
    ///
    /// - `timestamp`: Actual time this event was emitted (can be used for delta-time)
    /// - `target_timestamp`: Ideal time the frame should display on screen
    /// - `refresh_interval`: Monitor refresh period (e.g., 16.67ms for 60Hz)
    ///
    /// # Example Use
    ///
    /// Use `target_timestamp` to animate content that should be time-accurate.
    /// Render a manifold that interpolates based on the time delta.
    RequestFrame {
        timestamp: std::time::Instant,
        target_timestamp: std::time::Instant,
        refresh_interval: std::time::Duration,
    },
}

/// Commands sent from the Application to the Engine (frame rendering).
///
/// Application sends these to respond to frame requests. These are the output
/// of the application's render loop—typically sent in response to `EngineEvent::Data(RequestFrame)`.
///
/// # Message Contract
///
/// When the application sends a frame, it establishes a contract:
/// - **Precondition**: Application received a `RequestFrame` event
/// - **Action**: Engine buffers the manifold and renders it to the window
/// - **Postcondition**: Pixels are on screen at the next VSync
/// - **Blocking**: May block if buffer is full, but high-priority (won't be dropped)
///
/// # Generic Type Parameter
///
/// The pixel type `P` is kept for API compatibility but is unused in practice.
/// All rendering is done via manifolds that produce `Discrete` (packed RGBA u32 pixels).
pub enum AppData<P: pixelflow_graphics::Pixel> {
    /// Render a manifold (continuous surface) to the window.
    ///
    /// # Contract
    ///
    /// **Sender** (Application): Provides a color manifold that produces `Discrete` pixels.
    /// The manifold is evaluated at sequential x-coordinates and writes pixels to the framebuffer.
    ///
    /// **Receiver** (Engine): Materializes the manifold and presents it to the window.
    /// The manifold is evaluated fresh for each frame, so it can be animated or interactive.
    ///
    /// # Arguments
    ///
    /// - A `Manifold<Output = Discrete>` wrapped in `Arc<dyn ...>` for type erasure
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pixelflow_graphics::Color;
    /// use pixelflow_core::ManifoldExt;
    /// use std::sync::Arc;
    ///
    /// // Build a red rectangle
    /// let red = Color::Named(NamedColor::Red);
    /// let manifold = Arc::new(red) as Arc<dyn Manifold<Output = Discrete> + Send + Sync>;
    ///
    /// tx.send(Message::Data(AppData::RenderSurface(manifold)))?;
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - The manifold is evaluated for every frame, every pixel (no caching)
    /// - Use simple, closed-form expressions for good performance
    /// - The evaluation happens in the render thread; expensive computations will cause frame drops
    /// - Consider composing manifolds rather than using `map` callbacks
    ///
    /// # Type Erasure
    ///
    /// The manifold is stored as `dyn Manifold`, which erases the static type.
    /// This allows different manifold types to be sent without recompiling.
    /// Performance cost: no dynamic dispatch at the algebra level (monomorphization still works),
    /// but you lose some inlining opportunities across the Arc boundary.
    RenderSurface(
        std::sync::Arc<
            dyn pixelflow_core::Manifold<Output = pixelflow_core::Discrete> + Send + Sync,
        >,
    ),

    /// Render a manifold at u32 coordinates (pixel-aligned).
    ///
    /// # Contract
    ///
    /// **Sender** (Application): Provides a color manifold to be evaluated at pixel-aligned coordinates.
    ///
    /// **Receiver** (Engine): Materializes the manifold using integer u32 coordinates (no floating-point).
    /// This is identical to `RenderSurface` except for the coordinate type passed to evaluation.
    ///
    /// # When to Use
    ///
    /// - When your manifold is designed to work with integer coordinates (e.g., per-pixel patterns)
    /// - For pixel-perfect rendering without sub-pixel interpolation
    /// - When aliasing is acceptable or desirable
    ///
    /// # Example
    ///
    /// ```ignore
    /// // A checkerboard pattern using u32 coordinates
    /// let checkerboard = (X & 16u32).eq(0.0).select(black, white);
    /// tx.send(Message::Data(AppData::RenderSurfaceU32(checkerboard)))?;
    /// ```
    ///
    /// # Performance
    ///
    /// Slightly more efficient than `RenderSurface` if your manifold naturally works
    /// with integer coordinates, but the difference is usually negligible.
    RenderSurfaceU32(
        std::sync::Arc<
            dyn pixelflow_core::Manifold<Output = pixelflow_core::Discrete> + Send + Sync,
        >,
    ),

    /// Skip this frame (no rendering needed).
    ///
    /// # Contract
    ///
    /// **Sender** (Application): Indicates that the frame hasn't changed and doesn't need rendering.
    ///
    /// **Receiver** (Engine): Reuses the previous frame or pauses rendering.
    /// Useful for reducing power consumption when nothing is animating.
    ///
    /// # Example Use
    ///
    /// Terminal emulator receives `RequestFrame` but has no new content to render.
    /// Instead of rendering the same pixels again, it sends `Skipped`.
    ///
    /// # Note
    ///
    /// Not all platforms support frame skipping. The engine may ignore this
    /// and present a blank/black frame instead. Use only when you know the
    /// previous frame is still correct.
    Skipped,

    #[doc(hidden)]
    _Phantom(std::marker::PhantomData<P>),
}

impl<P: pixelflow_graphics::Pixel> std::fmt::Debug for AppData<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RenderSurface(_) => f.debug_tuple("RenderSurface").finish(),
            Self::RenderSurfaceU32(_) => f.debug_tuple("RenderSurfaceU32").finish(),
            Self::Skipped => f.debug_tuple("Skipped").finish(),
            Self::_Phantom(_) => f.debug_tuple("_Phantom").finish(),
        }
    }
}

/// Application trait that defines the logic.
pub trait Application {
    fn send(&self, event: EngineEvent) -> Result<(), crate::error::RuntimeError>;
}

impl Application
    for actor_scheduler::ActorHandle<EngineEventData, EngineEventControl, EngineEventManagement>
{
    fn send(&self, event: EngineEvent) -> Result<(), crate::error::RuntimeError> {
        let msg = match event {
            EngineEvent::Control(ctrl) => actor_scheduler::Message::Control(ctrl),
            EngineEvent::Management(mgmt) => actor_scheduler::Message::Management(mgmt),
            EngineEvent::Data(data) => actor_scheduler::Message::Data(data),
        };
        self.send(msg)
            .map_err(|e| crate::error::RuntimeError::EventSendError(e.to_string()))
    }
}

/// Application management commands (change title, etc.)
pub enum AppManagement {
    /// Configure the engine with initial settings (sent on startup).
    Configure(crate::config::EngineConfig),
    /// Register the application handle so the engine can send events back to the app.
    RegisterApp(std::sync::Arc<dyn Application + Send + Sync>),
    /// Request window creation.
    ///
    /// # Contract
    ///
    /// **Sender** (Application): Requests a new window with the specified settings.
    ///
    /// **Receiver** (Engine): Relays the request to the driver. Driver will create the window,
    /// assign an ID, and respond with WindowCreated event containing the assigned ID.
    CreateWindow(WindowDescriptor),
    SetTitle(String),
    ResizeRequest(u32, u32),
    CopyToClipboard(String),
    RequestPaste,
    SetCursorIcon(CursorIcon),
    Quit,
}

impl std::fmt::Debug for AppManagement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppManagement::Configure(config) => f.debug_tuple("Configure").field(config).finish(),
            AppManagement::RegisterApp(_) => f.debug_tuple("RegisterApp").field(&"<app>").finish(),
            AppManagement::CreateWindow(descriptor) => {
                f.debug_tuple("CreateWindow").field(descriptor).finish()
            }
            AppManagement::SetTitle(s) => f.debug_tuple("SetTitle").field(s).finish(),
            AppManagement::ResizeRequest(w, h) => {
                f.debug_tuple("ResizeRequest").field(&(w, h)).finish()
            }
            AppManagement::CopyToClipboard(s) => f.debug_tuple("CopyToClipboard").field(s).finish(),
            AppManagement::RequestPaste => f.write_str("RequestPaste"),
            AppManagement::SetCursorIcon(icon) => {
                f.debug_tuple("SetCursorIcon").field(icon).finish()
            }
            AppManagement::Quit => f.write_str("Quit"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CursorIcon {
    Default,
    Pointer,
    Text,
}

/// Descriptor for creating a new window.
#[derive(Debug, Clone)]
pub struct WindowDescriptor {
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub resizable: bool,
}

impl Default for WindowDescriptor {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            title: "PixelFlow".into(),
            resizable: true,
        }
    }
}
