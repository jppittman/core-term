// Test 1: Simple type (works currently)
use actor_scheduler::troupe;

struct SimpleActor;

troupe! {
    simple: SimpleActor,
}

fn main() {}
