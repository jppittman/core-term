use crate::api::private::{EngineActorHandle, EngineControl, EngineData};
use crate::api::public::AppManagement; // Use public re-export
use crate::platform::PlatformPixel;
use actor_scheduler::{Actor, ParkHint};
use std::sync::{Arc, Mutex};

/// A recorded message received by the MockEngine.
#[derive(Debug)] // Removed Clone as EngineData/Control might not be Clone
pub enum ReceivedMessage {
    Data(EngineData<PlatformPixel>),
    Control(EngineControl<PlatformPixel>),
    Management(AppManagement),
}

/// A mock engine that captures messages sent to it suitable for unit testing actors.
pub struct MockEngine {
    messages: Arc<Mutex<Vec<ReceivedMessage>>>,
    handle: EngineActorHandle<PlatformPixel>,
    _thread: Option<std::thread::JoinHandle<()>>,
}

impl MockEngine {
    /// Create a new MockEngine. Returns the engine instance (to inspect messages)
    /// and the handle (to pass to the actor under test).
    pub fn new() -> Self {
        let messages = Arc::new(Mutex::new(Vec::new()));

        let mut collector = MessageCollector {
            messages: messages.clone(),
        };

        // Create scheduler channels
        let (handle, mut scheduler) = actor_scheduler::create_actor::<
            EngineData<PlatformPixel>,
            EngineControl<PlatformPixel>,
            AppManagement,
        >(100, None);

        // Spawn background thread to process messages
        let thread = std::thread::spawn(move || {
            scheduler.run(&mut collector);
        });

        Self {
            messages,
            handle,
            _thread: Some(thread),
        }
    }

    pub fn handle(&self) -> EngineActorHandle<PlatformPixel> {
        self.handle.clone()
    }

    pub fn messages(&self) -> std::sync::MutexGuard<'_, Vec<ReceivedMessage>> {
        self.messages.lock().unwrap()
    }
}

// Collector Actor
struct MessageCollector {
    messages: Arc<Mutex<Vec<ReceivedMessage>>>,
}

impl Actor<EngineData<PlatformPixel>, EngineControl<PlatformPixel>, AppManagement>
    for MessageCollector
{
    fn handle_data(&mut self, msg: EngineData<PlatformPixel>) {
        self.messages
            .lock()
            .unwrap()
            .push(ReceivedMessage::Data(msg));
    }

    fn handle_control(&mut self, msg: EngineControl<PlatformPixel>) {
        self.messages
            .lock()
            .unwrap()
            .push(ReceivedMessage::Control(msg));
    }

    fn handle_management(&mut self, msg: AppManagement) {
        self.messages
            .lock()
            .unwrap()
            .push(ReceivedMessage::Management(msg));
    }

    fn park(&mut self, _hint: ParkHint) -> ParkHint {
        ParkHint::Wait
    }
}
