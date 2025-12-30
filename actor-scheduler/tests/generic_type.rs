// Test: troupe! macro should handle generic types
use actor_scheduler::{Actor, ActorTypes, ParkHint, TroupeActor};

struct Platform;
struct DriverActor<P> {
    _platform: std::marker::PhantomData<P>,
}

impl<P> Actor<(), (), ()> for DriverActor<P> {
    fn handle_data(&mut self, _: ()) {}
    fn handle_control(&mut self, _: ()) {}
    fn handle_management(&mut self, _: ()) {}
    fn park(&mut self, hint: ParkHint) -> ParkHint {
        hint
    }
}

// ActorTypes provides the message types without lifetime
impl ActorTypes for DriverActor<Platform> {
    type Data = ();
    type Control = ();
    type Management = ();
}

// TroupeActor just provides the constructor
impl<'a, Dir: 'a> TroupeActor<'a, Dir> for DriverActor<Platform> {
    fn new(_dir: &'a Dir) -> Self {
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
