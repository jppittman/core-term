use crate::api::private::{EngineActorHandle, EngineData};
use crate::display::messages::{DisplayControl, DisplayData, DisplayMgmt};
use crate::display::ops::{DriverEmit, DriverOut, PlatformOps};
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
pub struct PlatformActor<Ops: PlatformOps> {
    ops: Ops,
    engine_handle: EngineActorHandle,
    /// Reused across steps so a step's emits can be drained without reallocating the buffer.
    out: DriverOut,
}

impl<Ops: PlatformOps> PlatformActor<Ops> {
    pub fn new(ops: Ops, engine_handle: EngineActorHandle) -> Self {
        Self {
            ops,
            engine_handle,
            out: DriverOut::default(),
        }
    }

    /// Deliver everything the last step emitted, leaving the buffer empty for the next one.
    fn flush(&mut self) {
        for emit in self.out.emits.drain(..) {
            let data = match emit {
                DriverEmit::Event(event) => EngineData::FromDriver(event),
                DriverEmit::PresentComplete(window) => EngineData::PresentComplete(window),
            };
            self.engine_handle
                .send(Message::Data(data))
                .expect("failed to send engine event");
        }
    }
}

impl<Ops: PlatformOps> Actor<DisplayData, DisplayControl, DisplayMgmt> for PlatformActor<Ops> {
    fn handle_data(&mut self, msg: DisplayData) -> HandlerResult {
        let result = self.ops.handle_data(msg, &mut self.out);
        self.flush();
        result
    }

    fn handle_control(&mut self, msg: DisplayControl) -> HandlerResult {
        let result = self.ops.handle_control(msg, &mut self.out);
        self.flush();
        result
    }

    fn handle_management(&mut self, msg: DisplayMgmt) -> HandlerResult {
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
