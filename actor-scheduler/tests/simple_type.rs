// Test: troupe! macro with simple actor type
use actor_scheduler::{Actor, ActorTypes, TroupeActor, ActorStatus, SystemStatus, HandlerResult, HandlerError};

struct SimpleActor;

impl Actor<(), (), ()> for SimpleActor {
    fn handle_data(&mut self, _: ()) -> HandlerResult { Ok(()) }
    fn handle_control(&mut self, _: ()) -> HandlerResult { Ok(()) }
    fn handle_management(&mut self, _: ()) -> HandlerResult { Ok(()) }
    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

impl ActorTypes for SimpleActor {
    type Data = ();
    type Control = ();
    type Management = ();
}

impl<'a, Dir: 'a> TroupeActor<'a, Dir> for SimpleActor {
    fn new(_dir: &'a Dir) -> Self {
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
