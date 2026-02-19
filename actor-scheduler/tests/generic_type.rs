// Test: troupe! macro should handle generic types
use actor_scheduler::{
    Actor, ActorStatus, ActorTypes, HandlerError, HandlerResult, SystemStatus, TroupeActor,
};

struct Platform;
struct DriverActor<P> {
    _platform: std::marker::PhantomData<P>,
}

impl<P> Actor<(), (), ()> for DriverActor<P> {
    fn handle_data(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn handle_control(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn handle_management(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

// ActorTypes provides the message types without lifetime
impl ActorTypes for DriverActor<Platform> {
    type Data = ();
    type Control = ();
    type Management = ();
}

// TroupeActor just provides the constructor
impl<Dir> TroupeActor<Dir> for DriverActor<Platform> {
    fn new(_dir: Dir) -> Self {
        Self {
            _platform: std::marker::PhantomData,
        }
    }
}

actor_scheduler::troupe! {
    driver: DriverActor<Platform> [main],
}

fn main() {
    // Success if this compiles
    let _troupe = Troupe::new();
}
