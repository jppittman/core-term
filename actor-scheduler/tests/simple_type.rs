// Test: troupe! macro with simple actor type
use actor_scheduler::{Actor, ActorTypes, ActorStatus, TroupeActor};

struct SimpleActor;

impl Actor<(), (), ()> for SimpleActor {
    fn handle_data(&mut self, _: ()) -> Result<(), actor_scheduler::ActorError> {
        Ok(())
    }
    fn handle_control(&mut self, _: ()) -> Result<(), actor_scheduler::ActorError> {
        Ok(())
    }
    fn handle_management(&mut self, _: ()) -> Result<(), actor_scheduler::ActorError> {
        Ok(())
    }
    fn park(&mut self, hint: ActorStatus) -> ActorStatus {
        hint
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
