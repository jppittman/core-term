use super::messages::{DisplayControl, DisplayData, DisplayEvent, DisplayMgmt, Window};
use actor_scheduler::{ActorStatus, HandlerError, HandlerResult, SystemStatus};

/// One thing the driver emits toward the engine.
///
/// These are exactly the two `driver → engine` edges in the target topology
/// (`docs/designs/pixelflow-runtime-engine-mesh-migration.md` §3): the blocking input and
/// window-lifecycle event stream, and the window return that closes the `Present` loop.
#[derive(Debug)]
pub enum DriverEmit {
    /// An input or window-lifecycle event.
    Event(DisplayEvent),
    /// A window handed back after presentation, for reuse by the next frame.
    PresentComplete(Window),
}

/// Output word for a [`PlatformOps`] step.
///
/// One `Out` per step, matching `VsyncCoreOut`/`RasterCoreOut` — the struct's *fields* are the
/// ports, so a step never returns "sometimes a value, sometimes a list." `park` is the reason
/// this field is a `Vec` rather than an `Option`: draining the OS event queue yields N events.
///
/// `Vec` is not a per-frame allocation despite `park` running every frame: an empty `Vec` never
/// touches the heap, and the overwhelming majority of frames carry no input at all. Only a
/// frame where the user actually did something pays one small amortized growth.
#[derive(Debug, Default)]
pub struct DriverOut {
    pub emits: Vec<DriverEmit>,
}

impl DriverOut {
    /// Queue an input or window-lifecycle event.
    pub fn event(&mut self, event: DisplayEvent) {
        self.emits.push(DriverEmit::Event(event));
    }

    /// Queue a window return.
    pub fn present_complete(&mut self, window: Window) {
        self.emits.push(DriverEmit::PresentComplete(window));
    }
}

/// Backend-specific operations for the display platform.
///
/// Implementors own their platform resources (the X11 window, the `NSApplication`) and nothing
/// else — in particular **not** a handle to the engine. Outbound events are *returned* via
/// [`DriverOut`] rather than sent, so an implementation can be driven and observed with no
/// engine, no scheduler, and no channels in the loop. Performing the real sends is
/// `PlatformActor`'s job.
pub trait PlatformOps: Send + 'static {
    fn handle_data(&mut self, data: DisplayData, out: &mut DriverOut) -> HandlerResult;
    fn handle_control(&mut self, ctrl: DisplayControl, out: &mut DriverOut) -> HandlerResult;
    fn handle_management(&mut self, mgmt: DisplayMgmt, out: &mut DriverOut) -> HandlerResult;
    fn park(
        &mut self,
        status: SystemStatus,
        out: &mut DriverOut,
    ) -> Result<ActorStatus, HandlerError>;
}
