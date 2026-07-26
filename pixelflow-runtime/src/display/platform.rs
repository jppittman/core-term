use crate::api::private::{EngineActorHandle, EngineData};
use crate::display::messages::{DisplayControl, DisplayData, DisplayEvent, DisplayMgmt};
use crate::display::ops::{DriverEmit, DriverOut, PlatformOps};
use crate::display::window_keeper::{Presented, WindowKeeper};
use actor_scheduler::{Actor, ActorStatus, HandlerError, HandlerResult, Message, SystemStatus};

/// The Platform Trait.
/// Implementers must be an Actor that handles display messages.
pub trait Platform: Actor<DisplayData, DisplayControl, DisplayMgmt> + Send + 'static {}

/// A generic wrapper that turns any `PlatformOps` implementation into a `Platform` Actor.
///
/// This is the adapter half of the split described in
/// `docs/designs/pixelflow-runtime-engine-mesh-migration.md` §5.1: [`PlatformOps`] decides and
/// returns, this type performs the one effect it cannot — the actual send to the engine. The
/// engine handle lives here rather than inside the ops precisely so that the ops stay drivable
/// without an engine.
///
/// It also owns the window buffer, via [`WindowKeeper`] — the pull protocol of §8. The ops
/// report the geometry the OS chose and know nothing about who is drawing; this decides which
/// buffer exists and who is holding it. So two things never reach the ops at all:
/// [`DisplayMgmt::RequestWindow`], answered from the keeper, and the buffer coming back from a
/// blit, which returns to it.
pub struct PlatformActor<Ops: PlatformOps> {
    ops: Ops,
    engine_handle: EngineActorHandle,
    keeper: WindowKeeper,
    /// Reused across steps so a step's emits can be drained without reallocating the buffer.
    out: DriverOut,
}

impl<Ops: PlatformOps> PlatformActor<Ops> {
    pub fn new(ops: Ops, engine_handle: EngineActorHandle) -> Self {
        Self {
            ops,
            engine_handle,
            keeper: WindowKeeper::new(),
            out: DriverOut::default(),
        }
    }

    /// Deliver everything the last step emitted, then any grant that has come due.
    ///
    /// The grant goes last on purpose. All three things that can make one possible — a request
    /// arriving, the buffer coming back from a blit, a resize allocating a new one — are
    /// resolved by the time the emits are drained, so asking once here is enough and there is
    /// no path that changes the answer without passing through this.
    fn flush(&mut self) {
        self.drain_emits();
        if let Some(window) = self.keeper.pending_grant() {
            self.engine_handle
                .send(Message::Data(EngineData::WindowGranted(window)))
                .expect("failed to grant window");
        }
    }

    fn drain_emits(&mut self) {
        for emit in self.out.emits.drain(..) {
            match emit {
                // Window geometry is the keeper's business *and* the engine's: the keeper builds
                // the buffer, the engine tells the app its new size. Keeper first, so a
                // `RequestWindow` arriving right behind the event finds a buffer waiting.
                DriverEmit::Event(event) => {
                    if let DisplayEvent::WindowCreated { surface }
                    | DisplayEvent::Resized { surface } = event
                    {
                        self.keeper.surface_changed(surface);
                    }
                    self.engine_handle
                        .send(Message::Data(EngineData::FromDriver(event)))
                        .expect("failed to send engine event");
                }
                // Goes no further than here: the driver is the buffer's resting owner, so
                // `Present` was already the return and there is nothing to acknowledge.
                DriverEmit::Blitted(window) => self.keeper.rest(window),
            }
        }
    }
}

impl<Ops: PlatformOps> Actor<DisplayData, DisplayControl, DisplayMgmt> for PlatformActor<Ops> {
    fn handle_data(&mut self, msg: DisplayData) -> HandlerResult {
        let DisplayData::Present { window } = msg;
        // The one identity check the pull protocol keeps (§8): binding the kernel to the frame
        // makes the pixels right for *that buffer*, which cannot make an old-sized buffer right
        // for a window that has since changed shape.
        let result = match self.keeper.presented(window) {
            Presented::Blit(window) => self
                .ops
                .handle_data(DisplayData::Present { window }, &mut self.out),
            Presented::Superseded => Ok(()),
        };
        self.flush();
        result
    }

    fn handle_control(&mut self, msg: DisplayControl) -> HandlerResult {
        let result = self.ops.handle_control(msg, &mut self.out);
        self.flush();
        result
    }

    fn handle_management(&mut self, msg: DisplayMgmt) -> HandlerResult {
        // Recorded on the keeper, which `flush` then answers. The ops hold no buffer and never
        // see this.
        if matches!(msg, DisplayMgmt::RequestWindow) {
            self.keeper.request();
            self.flush();
            return Ok(());
        }
        let result = self.ops.handle_management(msg, &mut self.out);
        self.flush();
        result
    }

    fn park(&mut self, status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        let result = self.ops.park(status, &mut self.out);
        self.flush();
        result
    }
}

impl<Ops: PlatformOps> Platform for PlatformActor<Ops> {}
