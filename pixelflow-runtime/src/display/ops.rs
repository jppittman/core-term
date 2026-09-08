use super::messages::{DisplayControl, DisplayData, DisplayEvent, DisplayMgmt, Window};
use actor_scheduler::{ActorStatus, HandlerError, HandlerResult, SystemStatus};

/// One thing a [`PlatformOps`] step produced.
///
/// Not all of these leave the driver. `Event` is the blocking `driver → engine` edge in the
/// target topology; `Blitted` goes no further than `PlatformActor`, which puts the buffer back
/// at rest. There is no window *return* message any more: under the pull protocol the driver is
/// the buffer's resting owner, so `Present` **is** the return (§8 of
/// `docs/designs/pixelflow-runtime-engine-mesh-migration.md`).
#[derive(Debug)]
pub enum DriverEmit {
    /// An input or window-lifecycle event.
    Event(DisplayEvent),
    /// The buffer, after its pixels reached the screen. Handed back to the driver's keeper, not
    /// sent anywhere.
    Blitted(Window),
}

/// Output word for a [`PlatformOps`] step.
///
/// One `Out` per step, matching `VsyncCoreOut`/`RenderCoreOut` — the struct's *fields* are the
/// ports, so a step never returns "sometimes a value, sometimes a list." `handle_os` is the reason
/// this field is a `Vec` rather than an `Option`: draining the OS event queue yields N events.
///
/// `Vec` is not a per-frame allocation despite `handle_os` running every frame: an empty `Vec` never
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

    /// Hand the buffer back after blitting it.
    pub fn blitted(&mut self, window: Window) {
        self.emits.push(DriverEmit::Blitted(window));
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
    fn handle_os(
        &mut self,
        status: SystemStatus,
        out: &mut DriverOut,
    ) -> Result<ActorStatus, HandlerError>;
}
