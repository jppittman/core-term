//! Priority Channel - A multi-priority message passing system
//!
//! This crate provides a message scheduler with three priority levels:
//! - **Control**: Highest priority, unlimited drain
//! - **Management**: Medium priority, unlimited drain
//! - **Data**: Lowest priority, burst-limited with backpressure
//!
//! # Architecture
//!
//! The scheduler uses a "doorbell" pattern where:
//! 1. The receiver blocks on the Control channel
//! 2. Data messages send a Wake signal to unblock the receiver
//! 3. Priority processing drains Control → Management → Data
//!
//! # Example
//!
//! ```rust
//! use priority_channel::{PriorityReceiver, Message, SchedulerHandler};
//!
//! struct MyHandler;
//!
//! impl SchedulerHandler<String, String, String> for MyHandler {
//!     fn handle_data(&mut self, msg: String) {
//!         println!("Data: {}", msg);
//!     }
//!     fn handle_control(&mut self, msg: String) {
//!         println!("Control: {}", msg);
//!     }
//!     fn handle_management(&mut self, msg: String) {
//!         println!("Management: {}", msg);
//!     }
//! }
//!
//! let (tx, mut rx) = PriorityReceiver::<String, String, String>::new(10, 100);
//!
//! // Spawn receiver thread
//! std::thread::spawn(move || {
//!     let mut handler = MyHandler;
//!     rx.run(&mut handler);
//! });
//!
//! // Send messages from any thread
//! tx.send(Message::Data("low priority data".to_string())).unwrap();
//! tx.send(Message::Control("high priority control".to_string())).unwrap();
//! ```

use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TryRecvError};

/// The types of messages supported by the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message<D, C, M> {
    Data(D),
    Control(C),
    Management(M),
}

/// Internal wrapper for the control channel to support auto-waking.
#[derive(Debug, Clone)]
enum Signal<C> {
    User(C),
    Wake,
}

/// A trait for handling scheduled messages.
/// Implement this to define the behavior of your actor/worker.
pub trait SchedulerHandler<D, C, M> {
    fn handle_data(&mut self, msg: D);
    fn handle_control(&mut self, msg: C);
    fn handle_management(&mut self, msg: M);
}

/// A unified sender handle that routes messages to the appropriate priority lane.
#[derive(Clone)]
pub struct PrioritySender<D, C, M> {
    // Data is Bounded (SyncSender) for backpressure
    tx_data: SyncSender<D>,
    // Control/Mgmt are Unbounded (Sender) so they never block the sender
    tx_control: Sender<Signal<C>>,
    tx_mgmt: Sender<M>,
}

impl<D, C, M> PrioritySender<D, C, M> {
    /// Sends a message to the appropriate priority lane and wakes the scheduler.
    ///
    /// # Blocking Behavior
    /// - `Data`: **BLOCKS** if the data buffer is full (Backpressure).
    /// - `Control/Mgmt`: Non-blocking (Unbounded).
    ///
    /// # Errors
    /// Returns `Err` if the receiver has been dropped (disconnected).
    pub fn send(&self, msg: Message<D, C, M>) -> Result<(), mpsc::SendError<()>> {
        match msg {
            Message::Data(d) => {
                self.tx_data.send(d).map_err(|_| mpsc::SendError(()))?;
                let _ = self.tx_control.send(Signal::Wake);
            }
            Message::Control(c) => {
                self.tx_control
                    .send(Signal::User(c))
                    .map_err(|_| mpsc::SendError(()))?;
            }
            Message::Management(m) => {
                self.tx_mgmt.send(m).map_err(|_| mpsc::SendError(()))?;
                let _ = self.tx_control.send(Signal::Wake);
            }
        };
        Ok(())
    }
}

/// The receiver side that implements the priority scheduling logic.
pub struct PriorityReceiver<D, C, M> {
    rx_data: Receiver<D>,
    rx_control: Receiver<Signal<C>>,
    rx_mgmt: Receiver<M>,
    data_burst_limit: usize,
}

impl<D, C, M> PriorityReceiver<D, C, M> {
    /// Create a new priority channel.
    ///
    /// # Arguments
    /// * `data_burst_limit` - Maximum data messages to process per wake cycle
    /// * `data_buffer_size` - Size of bounded data buffer (backpressure threshold)
    ///
    /// # Returns
    /// Returns `(sender, receiver)` tuple. The sender can be cloned and shared.
    pub fn new(data_burst_limit: usize, data_buffer_size: usize) -> (PrioritySender<D, C, M>, Self) {
        let (tx_data, rx_data) = mpsc::sync_channel(data_buffer_size);
        let (tx_control, rx_control) = mpsc::channel();
        let (tx_mgmt, rx_mgmt) = mpsc::channel();

        let sender = PrioritySender {
            tx_data,
            tx_control,
            tx_mgmt,
        };

        let receiver = PriorityReceiver {
            rx_data,
            rx_control,
            rx_mgmt,
            data_burst_limit,
        };

        (sender, receiver)
    }

    /// The Main Scheduler Loop.
    /// Blocks on the Control pipe. Prioritizes Control > Management > Data.
    ///
    /// # Arguments
    /// * `handler` - Implementation of `SchedulerHandler` trait
    ///
    /// This method runs forever until all senders are dropped.
    pub fn run<H>(&mut self, handler: &mut H)
    where
        H: SchedulerHandler<D, C, M>,
    {
        loop {
            // 1. Block on Control Pipe (The "Doorbell")
            let first_signal = match self.rx_control.recv() {
                Ok(s) => s,
                Err(_) => return, // All senders disconnected
            };

            match first_signal {
                Signal::User(c) => handler.handle_control(c),
                Signal::Wake => {}
            }

            // 2. Priority Processing Loop
            let mut keep_working = true;
            while keep_working {
                keep_working = false;

                // A. Control (Highest Priority - Unlimited Drain)
                while let Ok(signal) = self.rx_control.try_recv() {
                    match signal {
                        Signal::User(c) => {
                            handler.handle_control(c);
                            keep_working = true;
                        }
                        Signal::Wake => {}
                    }
                }

                // B. Management (Medium Priority - Unlimited Drain)
                while let Ok(msg) = self.rx_mgmt.try_recv() {
                    handler.handle_management(msg);
                    keep_working = true;
                }

                // C. Data (Low Priority - Burst Limited)
                let mut data_count = 0;
                while data_count < self.data_burst_limit {
                    match self.rx_data.try_recv() {
                        Ok(msg) => {
                            handler.handle_data(msg);
                            data_count += 1;
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => return,
                    }
                }

                if data_count >= self.data_burst_limit {
                    keep_working = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    struct TestHandler {
        log: Arc<Mutex<Vec<String>>>,
    }

    impl SchedulerHandler<String, String, String> for TestHandler {
        fn handle_data(&mut self, msg: String) {
            self.log.lock().unwrap().push(format!("Data: {}", msg));
        }
        fn handle_control(&mut self, msg: String) {
            self.log.lock().unwrap().push(format!("Ctrl: {}", msg));
        }
        fn handle_management(&mut self, msg: String) {
            self.log.lock().unwrap().push(format!("Mgmt: {}", msg));
        }
    }

    #[test]
    fn test_priority_ordering() {
        let (tx, mut rx) = PriorityReceiver::new(2, 10);
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_clone = log.clone();

        let handle = thread::spawn(move || {
            let mut handler = TestHandler { log: log_clone };
            rx.run(&mut handler);
        });

        // Send messages in mixed order
        tx.send(Message::Data("1".to_string())).unwrap();
        tx.send(Message::Management("M".to_string())).unwrap();
        tx.send(Message::Data("2".to_string())).unwrap();
        tx.send(Message::Control("C".to_string())).unwrap();
        tx.send(Message::Data("3".to_string())).unwrap();

        thread::sleep(Duration::from_millis(50));

        // Drop sender to close channels and stop run()
        drop(tx);
        handle.join().unwrap();

        let messages = log.lock().unwrap();
        assert!(messages.len() > 0, "Should have processed messages");

        // Control should be processed before lower priority messages
        let ctrl_idx = messages.iter().position(|s| s.contains("Ctrl")).unwrap();
        let data1_idx = messages.iter().position(|s| s.contains("Data: 1")).unwrap();

        assert!(
            ctrl_idx < data1_idx,
            "Control should be processed before Data that was sent earlier"
        );
    }

    #[test]
    fn test_backpressure() {
        let (tx, mut rx) = PriorityReceiver::new(2, 1); // Buffer size 1, burst limit 2
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_clone = log.clone();

        thread::spawn(move || {
            let mut handler = TestHandler { log: log_clone };
            rx.run(&mut handler);
        });

        let tx_clone = tx.clone();
        let send_thread = thread::spawn(move || {
            tx_clone.send(Message::Data("1".to_string())).unwrap();
            tx_clone.send(Message::Data("2".to_string())).unwrap(); // Should block
            tx_clone.send(Message::Data("3".to_string())).unwrap();
        });

        thread::sleep(Duration::from_millis(100));
        drop(tx);
        send_thread.join().unwrap();

        thread::sleep(Duration::from_millis(50));
        let messages = log.lock().unwrap();
        assert_eq!(messages.len(), 3, "All messages should be processed");
    }

    #[test]
    fn test_trait_handler() {
        struct CountingHandler {
            data_count: usize,
            ctrl_count: usize,
            mgmt_count: usize,
        }

        impl SchedulerHandler<i32, String, bool> for CountingHandler {
            fn handle_data(&mut self, _: i32) {
                self.data_count += 1;
            }
            fn handle_control(&mut self, _: String) {
                self.ctrl_count += 1;
            }
            fn handle_management(&mut self, _: bool) {
                self.mgmt_count += 1;
            }
        }

        let (tx, mut rx) = PriorityReceiver::new(10, 100);

        let handle = thread::spawn(move || {
            let mut handler = CountingHandler {
                data_count: 0,
                ctrl_count: 0,
                mgmt_count: 0,
            };
            rx.run(&mut handler);
            handler
        });

        tx.send(Message::Data(1)).unwrap();
        tx.send(Message::Data(2)).unwrap();
        tx.send(Message::Control("test".to_string())).unwrap();
        tx.send(Message::Management(true)).unwrap();

        thread::sleep(Duration::from_millis(50));
        drop(tx);

        let handler = handle.join().unwrap();
        assert_eq!(handler.data_count, 2);
        assert_eq!(handler.ctrl_count, 1);
        assert_eq!(handler.mgmt_count, 1);
    }
}
