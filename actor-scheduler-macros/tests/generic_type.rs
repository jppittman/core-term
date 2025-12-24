// Test: troupe! macro should handle generic types
use actor_scheduler::{Actor, ParkHint, TroupeActor};

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

impl<'a, Dir> TroupeActor<'a, Dir> for DriverActor<Platform> {
    type Data = ();
    type Control = ();
    type Management = ();
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
