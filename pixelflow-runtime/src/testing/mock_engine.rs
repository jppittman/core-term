use crate::api::private::{EngineActorHandle, EngineControl, EngineData};
use crate::api::public::AppManagement; // Use public re-export
use actor_scheduler::{Actor, ActorStatus, HandlerError, HandlerResult, SystemStatus};
use std::sync::{Arc, Mutex};

/// A recorded message received by the MockEngine.
#[derive(Debug)]
pub enum ReceivedMessage {
    Data(EngineData),
    Control(EngineControl),
    Management(AppManagement),
}

/// A mock engine that captures messages sent to it suitable for unit testing actors.
pub struct MockEngine {
    messages: Arc<Mutex<Vec<ReceivedMessage>>>,
    handle: EngineActorHandle,
    _thread: Option<std::thread::JoinHandle<()>>,
}

impl MockEngine {
    /// Create a new MockEngine. Returns the engine instance (to inspect messages)
    /// and the handle (to pass to the actor under test).
    #[must_use]
    pub fn new() -> Self {
        let messages = Arc::new(Mutex::new(Vec::new()));

        let mut collector = MessageCollector {
            messages: messages.clone(),
        };

        // Create scheduler channels
        let (handle, mut scheduler) =
            actor_scheduler::create_actor::<EngineData, EngineControl, AppManagement>(100, None);

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

    #[must_use]
    pub fn handle(&self) -> EngineActorHandle {
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

impl Actor<EngineData, EngineControl, AppManagement> for MessageCollector {
    fn handle_data(&mut self, msg: EngineData) -> HandlerResult {
        self.messages
            .lock()
            .unwrap()
            .push(ReceivedMessage::Data(msg));
        Ok(())
    }

    fn handle_control(&mut self, msg: EngineControl) -> HandlerResult {
        self.messages
            .lock()
            .unwrap()
            .push(ReceivedMessage::Control(msg));
        Ok(())
    }

    fn handle_management(&mut self, msg: AppManagement) -> HandlerResult {
        self.messages
            .lock()
            .unwrap()
            .push(ReceivedMessage::Management(msg));
        Ok(())
    }

    fn park(&mut self, _status: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}
