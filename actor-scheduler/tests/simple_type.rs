// Test: troupe! macro with simple actor type
use actor_scheduler::{
    Actor, ActorStatus, ActorTypes, HandlerError, HandlerResult, SystemStatus, TroupeActor,
};

struct SimpleActor;

impl Actor<(), (), ()> for SimpleActor {
    type Error = String;
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

impl ActorTypes for SimpleActor {
    type Data = ();
    type Control = ();
    type Management = ();
}

impl<Dir> TroupeActor<Dir> for SimpleActor {
    fn new(_dir: Dir) -> Self {
        Self
    }
}

actor_scheduler::troupe! {
    simple: SimpleActor [main],
}

fn main() {
    // Success if this compiles
    let _troupe = Troupe::new();
}
