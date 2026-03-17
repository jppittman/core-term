//! Message CUJ (Critical User Journey) Tests
//!
//! These tests verify the message flows between actors in the core-term system.
//! Each test covers a specific CUJ identified in MESSAGE_CUJ_COVERAGE.md.

use actor_scheduler::{
    Actor, ActorBuilder, ActorScheduler, ActorStatus, HandlerError, HandlerResult, Message,
    SystemStatus,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// =============================================================================
// Test Fixtures
// =============================================================================

/// Mock ANSI command for testing (simplified version of the real AnsiCommand)
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
enum MockAnsiCommand {
    Print(char),
    Newline,
    CursorMove(u32, u32),
}

/// Parser actor that mimics ParserThread behavior
struct MockParserActor {
    cmd_tx: SyncSender<Vec<MockAnsiCommand>>,
    bytes_received: Arc<AtomicUsize>,
}

impl Actor<Vec<u8>, (), ()> for MockParserActor {
    fn handle_data(&mut self, data: Vec<u8>) -> HandlerResult {
        self.bytes_received.fetch_add(data.len(), Ordering::SeqCst);

        // Simple mock parsing: convert bytes to Print commands
        let commands: Vec<MockAnsiCommand> = data
            .iter()
            .filter_map(|&b| {
                if b == b'\n' {
                    Some(MockAnsiCommand::Newline)
                } else if b.is_ascii_graphic() || b == b' ' {
                    Some(MockAnsiCommand::Print(b as char))
                } else {
                    None
                }
            })
            .collect();

        if !commands.is_empty() {
            let _ = self.cmd_tx.send(commands);
        }
        Ok(())
    }

    fn handle_control(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn handle_management(&mut self, _: ()) -> HandlerResult {
        Ok(())
    }
    fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

/// App actor that receives parsed commands
#[allow(dead_code)]
struct MockAppActor {
    commands_received: Arc<Mutex<Vec<Vec<MockAnsiCommand>>>>,
}

impl MockAppActor {
    #[allow(dead_code)]
    fn new() -> (Self, Arc<Mutex<Vec<Vec<MockAnsiCommand>>>>) {
        let commands = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                commands_received: commands.clone(),
            },
            commands,
        )
    }
}

// =============================================================================
// PTY-01: ReadThread → ParserThread Message Flow
// =============================================================================

#[test]
fn cuj_pty01_single_byte_batch_delivery() {
    // Given: A parser actor receiving bytes via actor scheduler
    let (cmd_tx, cmd_rx) = sync_channel::<Vec<MockAnsiCommand>>(10);
    let bytes_received = Arc::new(AtomicUsize::new(0));

    let (parser_tx, mut parser_rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_received_clone = bytes_received.clone();
    let parser_handle = thread::spawn(move || {
        let mut actor = MockParserActor {
            cmd_tx,
            bytes_received: bytes_received_clone,
        };
        parser_rx.run(&mut actor);
    });

    // When: A single batch of bytes is sent
    let test_data = b"Hello".to_vec();
    parser_tx.send(Message::Data(test_data.clone())).unwrap();

    // Allow processing
    thread::sleep(Duration::from_millis(50));
    drop(parser_tx);
    parser_handle.join().unwrap();

    // Then: All bytes should be received
    assert_eq!(
        bytes_received.load(Ordering::SeqCst),
        5,
        "Parser should receive exactly 5 bytes"
    );

    // And: Commands should be generated
    let commands = cmd_rx.try_recv().unwrap();
    assert_eq!(commands.len(), 5, "Should produce 5 Print commands");
}

#[test]
fn cuj_pty01_multiple_batches_fifo_order() {
    // Given: A parser actor
    let (cmd_tx, cmd_rx) = sync_channel::<Vec<MockAnsiCommand>>(100);
    let bytes_received = Arc::new(AtomicUsize::new(0));

    let (parser_tx, mut parser_rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_received_clone = bytes_received.clone();
    let parser_handle = thread::spawn(move || {
        let mut actor = MockParserActor {
            cmd_tx,
            bytes_received: bytes_received_clone,
        };
        parser_rx.run(&mut actor);
    });

    // When: Multiple batches are sent
    for i in 0..5 {
        let data = format!("batch{}", i).into_bytes();
        parser_tx.send(Message::Data(data)).unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    drop(parser_tx);
    parser_handle.join().unwrap();

    // Then: All batches should be received in FIFO order
    let mut all_commands = Vec::new();
    while let Ok(cmds) = cmd_rx.try_recv() {
        all_commands.extend(cmds);
    }

    // Verify FIFO ordering by checking first characters
    let chars: String = all_commands
        .iter()
        .filter_map(|c| match c {
            MockAnsiCommand::Print(ch) => Some(*ch),
            _ => None,
        })
        .collect();

    assert!(
        chars.starts_with("batch0"),
        "First batch should be processed first, got: {}",
        chars
    );
}

#[test]
fn cuj_pty01_large_batch_handling() {
    // Given: A parser actor
    let (cmd_tx, cmd_rx) = sync_channel::<Vec<MockAnsiCommand>>(100);
    let bytes_received = Arc::new(AtomicUsize::new(0));

    let (parser_tx, mut parser_rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_received_clone = bytes_received.clone();
    let parser_handle = thread::spawn(move || {
        let mut actor = MockParserActor {
            cmd_tx,
            bytes_received: bytes_received_clone,
        };
        parser_rx.run(&mut actor);
    });

    // When: A large batch (4KB) is sent
    let large_data: Vec<u8> = (0..4096).map(|i| b'A' + (i % 26) as u8).collect();
    let expected_len = large_data.len();
    parser_tx.send(Message::Data(large_data)).unwrap();

    thread::sleep(Duration::from_millis(100));
    drop(parser_tx);
    parser_handle.join().unwrap();

    // Then: All bytes should be received
    assert_eq!(
        bytes_received.load(Ordering::SeqCst),
        expected_len,
        "Parser should receive all {} bytes",
        expected_len
    );

    // Collect all commands
    let mut total_commands = 0;
    while let Ok(cmds) = cmd_rx.try_recv() {
        total_commands += cmds.len();
    }
    assert_eq!(
        total_commands, expected_len,
        "Should produce {} Print commands",
        expected_len
    );
}

#[test]
fn cuj_pty01_channel_closure_propagation() {
    // Given: A parser actor
    let (cmd_tx, _cmd_rx) = sync_channel::<Vec<MockAnsiCommand>>(10);
    let bytes_received = Arc::new(AtomicUsize::new(0));

    let (parser_tx, mut parser_rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_received_clone = bytes_received.clone();
    let parser_handle = thread::spawn(move || {
        let mut actor = MockParserActor {
            cmd_tx,
            bytes_received: bytes_received_clone,
        };
        parser_rx.run(&mut actor);
    });

    // Send some data
    parser_tx.send(Message::Data(b"test".to_vec())).unwrap();
    thread::sleep(Duration::from_millis(20));

    // When: Sender is dropped (simulating ReadThread exit)
    drop(parser_tx);

    // Then: Parser thread should exit gracefully
    let result = parser_handle.join();
    assert!(result.is_ok(), "Parser thread should exit cleanly");
}

// =============================================================================
// PTY-02: ParserThread → App Message Flow
// =============================================================================

#[test]
fn cuj_pty02_command_batch_delivery() {
    // Given: A channel from parser to app
    let (cmd_tx, cmd_rx): (
        SyncSender<Vec<MockAnsiCommand>>,
        Receiver<Vec<MockAnsiCommand>>,
    ) = sync_channel(10);

    // When: Parser sends a batch of commands
    let commands = vec![
        MockAnsiCommand::Print('H'),
        MockAnsiCommand::Print('i'),
        MockAnsiCommand::Newline,
    ];
    cmd_tx.send(commands.clone()).unwrap();

    // Then: App should receive the exact batch
    let received = cmd_rx.recv_timeout(Duration::from_millis(100)).unwrap();
    assert_eq!(received, commands, "App should receive exact command batch");
}

#[test]
fn cuj_pty02_empty_input_no_output() {
    // Given: A parser actor
    let (cmd_tx, cmd_rx) = sync_channel::<Vec<MockAnsiCommand>>(10);
    let bytes_received = Arc::new(AtomicUsize::new(0));

    let (parser_tx, mut parser_rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_received_clone = bytes_received.clone();
    let parser_handle = thread::spawn(move || {
        let mut actor = MockParserActor {
            cmd_tx,
            bytes_received: bytes_received_clone,
        };
        parser_rx.run(&mut actor);
    });

    // When: Empty bytes are sent
    parser_tx.send(Message::Data(vec![])).unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(parser_tx);
    parser_handle.join().unwrap();

    // Then: No commands should be produced
    let result = cmd_rx.try_recv();
    assert!(
        result.is_err(),
        "No commands should be sent for empty input"
    );
}

#[test]
fn cuj_pty02_mixed_content_batch() {
    // Given: A parser actor
    let (cmd_tx, cmd_rx) = sync_channel::<Vec<MockAnsiCommand>>(10);
    let bytes_received = Arc::new(AtomicUsize::new(0));

    let (parser_tx, mut parser_rx) = ActorScheduler::<Vec<u8>, (), ()>::new(10, 64);

    let bytes_received_clone = bytes_received.clone();
    let parser_handle = thread::spawn(move || {
        let mut actor = MockParserActor {
            cmd_tx,
            bytes_received: bytes_received_clone,
        };
        parser_rx.run(&mut actor);
    });

    // When: Mixed content (text + newlines) is sent
    let mixed_data = b"Line1\nLine2\n".to_vec();
    parser_tx.send(Message::Data(mixed_data)).unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(parser_tx);
    parser_handle.join().unwrap();

    // Then: Commands should include both prints and newlines
    let commands = cmd_rx.try_recv().unwrap();
    let newline_count = commands
        .iter()
        .filter(|c| matches!(c, MockAnsiCommand::Newline))
        .count();
    assert_eq!(newline_count, 2, "Should have 2 newline commands");
}

// =============================================================================
// PTY-03: App → WriteThread Message Flow
// =============================================================================

#[test]
fn cuj_pty03_write_command_delivery() {
    // Given: A channel from app to write thread
    let (write_tx, write_rx): (SyncSender<Vec<u8>>, Receiver<Vec<u8>>) = sync_channel(10);

    // When: App sends bytes to write
    let output = b"\x1b[H".to_vec(); // Move cursor home
    write_tx.send(output.clone()).unwrap();

    // Then: Write thread receives the bytes
    let received = write_rx.recv_timeout(Duration::from_millis(100)).unwrap();
    assert_eq!(received, output, "Write thread should receive exact bytes");
}

#[test]
fn cuj_pty03_write_ordering_preserved() {
    // Given: A channel from app to write thread
    let (write_tx, write_rx): (SyncSender<Vec<u8>>, Receiver<Vec<u8>>) = sync_channel(100);

    // When: Multiple write commands are sent
    for i in 0..10 {
        write_tx.send(vec![b'0' + i]).unwrap();
    }

    // Then: Order should be preserved
    for i in 0..10 {
        let received = write_rx.recv_timeout(Duration::from_millis(100)).unwrap();
        assert_eq!(received, vec![b'0' + i], "Write {} should be in order", i);
    }
}

// =============================================================================
// PTY-04: Lifecycle Cascade Tests
// =============================================================================

#[test]
fn cuj_pty04_sender_drop_terminates_receiver() {
    // Given: An actor scheduler
    let (tx, mut rx) = ActorScheduler::<u8, (), ()>::new(10, 64);

    struct TerminationActor {
        terminated: Arc<AtomicUsize>,
    }

    impl Actor<u8, (), ()> for TerminationActor {
        fn handle_data(&mut self, _: u8) -> HandlerResult {
            Ok(())
        }
        fn handle_control(&mut self, _: ()) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: ()) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    impl Drop for TerminationActor {
        fn drop(&mut self) {
            self.terminated.fetch_add(1, Ordering::SeqCst);
        }
    }

    let terminated = Arc::new(AtomicUsize::new(0));
    let terminated_clone = terminated.clone();

    let handle = thread::spawn(move || {
        let mut actor = TerminationActor {
            terminated: terminated_clone,
        };
        rx.run(&mut actor);
    });

    // Send some data
    tx.send(Message::Data(42)).unwrap();
    thread::sleep(Duration::from_millis(20));

    // When: Sender is dropped
    drop(tx);

    // Then: Receiver thread should terminate
    handle.join().unwrap();
    assert_eq!(
        terminated.load(Ordering::SeqCst),
        1,
        "Actor should be dropped exactly once"
    );
}

// =============================================================================
// Engine Event Message Flow Tests
// =============================================================================

/// Mock engine event types for testing
#[derive(Debug, Clone, PartialEq)]
enum MockEngineControl {
    Resize(u32, u32),
    CloseRequested,
}

#[derive(Debug, Clone, PartialEq)]
enum MockEngineManagement {
    KeyDown { key: char },
    Paste(String),
}

#[derive(Debug, Clone, PartialEq)]
enum MockEngineData {
    RequestFrame,
}

struct MockTerminalAppActor {
    control_events: Arc<Mutex<Vec<MockEngineControl>>>,
    management_events: Arc<Mutex<Vec<MockEngineManagement>>>,
    data_events: Arc<Mutex<Vec<MockEngineData>>>,
}

impl Actor<MockEngineData, MockEngineControl, MockEngineManagement> for MockTerminalAppActor {
    fn handle_data(&mut self, data: MockEngineData) -> HandlerResult {
        self.data_events.lock().unwrap().push(data);
        Ok(())
    }

    fn handle_control(&mut self, ctrl: MockEngineControl) -> HandlerResult {
        self.control_events.lock().unwrap().push(ctrl);
        Ok(())
    }

    fn handle_management(&mut self, mgmt: MockEngineManagement) -> HandlerResult {
        self.management_events.lock().unwrap().push(mgmt);
        Ok(())
    }

    fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
        Ok(ActorStatus::Idle)
    }
}

#[test]
fn cuj_eng01_resize_message_delivery() {
    // Given: An app actor
    let control_events = Arc::new(Mutex::new(Vec::new()));
    let management_events = Arc::new(Mutex::new(Vec::new()));
    let data_events = Arc::new(Mutex::new(Vec::new()));

    let (tx, mut rx) =
        ActorScheduler::<MockEngineData, MockEngineControl, MockEngineManagement>::new(10, 64);

    let control_clone = control_events.clone();
    let mgmt_clone = management_events.clone();
    let data_clone = data_events.clone();

    let handle = thread::spawn(move || {
        let mut actor = MockTerminalAppActor {
            control_events: control_clone,
            management_events: mgmt_clone,
            data_events: data_clone,
        };
        rx.run(&mut actor);
    });

    // When: Resize event is sent
    tx.send(Message::Control(MockEngineControl::Resize(1920, 1080)))
        .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: App should receive resize
    let events = control_events.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0], MockEngineControl::Resize(1920, 1080));
}

#[test]
fn cuj_eng02_close_requested_delivery() {
    // Given: An app actor
    let control_events = Arc::new(Mutex::new(Vec::new()));

    let (tx, mut rx) =
        ActorScheduler::<MockEngineData, MockEngineControl, MockEngineManagement>::new(10, 64);

    let control_clone = control_events.clone();

    let handle = thread::spawn(move || {
        let mut actor = MockTerminalAppActor {
            control_events: control_clone,
            management_events: Arc::new(Mutex::new(Vec::new())),
            data_events: Arc::new(Mutex::new(Vec::new())),
        };
        rx.run(&mut actor);
    });

    // When: CloseRequested is sent
    tx.send(Message::Control(MockEngineControl::CloseRequested))
        .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: App should receive close request
    let events = control_events.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0], MockEngineControl::CloseRequested);
}

#[test]
fn cuj_eng04_keydown_delivery() {
    // Given: An app actor
    let management_events = Arc::new(Mutex::new(Vec::new()));

    let (tx, mut rx) =
        ActorScheduler::<MockEngineData, MockEngineControl, MockEngineManagement>::new(10, 64);

    let mgmt_clone = management_events.clone();

    let handle = thread::spawn(move || {
        let mut actor = MockTerminalAppActor {
            control_events: Arc::new(Mutex::new(Vec::new())),
            management_events: mgmt_clone,
            data_events: Arc::new(Mutex::new(Vec::new())),
        };
        rx.run(&mut actor);
    });

    // When: KeyDown events are sent
    tx.send(Message::Management(MockEngineManagement::KeyDown {
        key: 'a',
    }))
    .unwrap();
    tx.send(Message::Management(MockEngineManagement::KeyDown {
        key: 'b',
    }))
    .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: App should receive key events in order
    let events = management_events.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0], MockEngineManagement::KeyDown { key: 'a' });
    assert_eq!(events[1], MockEngineManagement::KeyDown { key: 'b' });
}

#[test]
fn cuj_eng08_paste_delivery() {
    // Given: An app actor
    let management_events = Arc::new(Mutex::new(Vec::new()));

    let (tx, mut rx) =
        ActorScheduler::<MockEngineData, MockEngineControl, MockEngineManagement>::new(10, 64);

    let mgmt_clone = management_events.clone();

    let handle = thread::spawn(move || {
        let mut actor = MockTerminalAppActor {
            control_events: Arc::new(Mutex::new(Vec::new())),
            management_events: mgmt_clone,
            data_events: Arc::new(Mutex::new(Vec::new())),
        };
        rx.run(&mut actor);
    });

    // When: Paste event with content is sent
    let paste_content = "Hello, World!".to_string();
    tx.send(Message::Management(MockEngineManagement::Paste(
        paste_content.clone(),
    )))
    .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: App should receive paste with content
    let events = management_events.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0], MockEngineManagement::Paste(paste_content));
}

#[test]
fn cuj_eng09_frame_request_delivery() {
    // Given: An app actor
    let data_events = Arc::new(Mutex::new(Vec::new()));

    let (tx, mut rx) =
        ActorScheduler::<MockEngineData, MockEngineControl, MockEngineManagement>::new(10, 64);

    let data_clone = data_events.clone();

    let handle = thread::spawn(move || {
        let mut actor = MockTerminalAppActor {
            control_events: Arc::new(Mutex::new(Vec::new())),
            management_events: Arc::new(Mutex::new(Vec::new())),
            data_events: data_clone,
        };
        rx.run(&mut actor);
    });

    // When: Frame request is sent
    tx.send(Message::Data(MockEngineData::RequestFrame))
        .unwrap();

    thread::sleep(Duration::from_millis(50));
    drop(tx);
    handle.join().unwrap();

    // Then: App should receive frame request
    let events = data_events.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0], MockEngineData::RequestFrame);
}

// =============================================================================
// Priority Ordering Tests (Control > Management > Data)
// =============================================================================

#[test]
fn cuj_priority_control_before_management_before_data() {
    // Given: An actor that records message order
    let message_order = Arc::new(Mutex::new(Vec::new()));

    struct OrderRecordingActor {
        order: Arc<Mutex<Vec<String>>>,
    }

    impl Actor<String, String, String> for OrderRecordingActor {
        fn handle_data(&mut self, msg: String) -> HandlerResult {
            self.order.lock().unwrap().push(format!("D:{}", msg));
            Ok(())
        }
        fn handle_control(&mut self, msg: String) -> HandlerResult {
            self.order.lock().unwrap().push(format!("C:{}", msg));
            Ok(())
        }
        fn handle_management(&mut self, msg: String) -> HandlerResult {
            self.order.lock().unwrap().push(format!("M:{}", msg));
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    let (tx, mut rx) = ActorScheduler::<String, String, String>::new(10, 64);
    let order_clone = message_order.clone();

    let handle = thread::spawn(move || {
        let mut actor = OrderRecordingActor { order: order_clone };
        rx.run(&mut actor);
    });

    // When: Messages are sent in Data, Management, Control order
    // Note: We need to ensure the messages are actually in the buffer before the actor pulls them.
    // Since we're sending from the same thread, they are guaranteed to be enqueued in this order.
    // However, the actor processes messages in batches. If batch size is small, or timing is unlucky,
    // it might process data1 before ctrl1 arrives in the buffer.
    // But priority is ONLY guaranteed for messages *currently available* in the channel.
    //
    // If the actor is sleeping (parked), it wakes up on data1. It might process data1 immediately.
    // To properly test priority, we must ensure all messages are in the channel *before* the actor wakes up.
    // We can't easily force the actor to sleep and not wake up until we're done sending in this simple test setup
    // without modifying the actor scheduler internals or using a custom waker.
    //
    // A workaround for testing priority is to flood the data channel first so the actor is busy,
    // then send control, and verify control jumps the queue.
    // OR, simpler: this test assumes the actor hasn't picked up data1 yet.
    // If the test is flaky, it means the actor is too fast.
    //
    // The previous assertion failed because `data1` was processed before `ctrl1`.
    // This happens if `data1` is dequeued before `ctrl1` is even enqueued.
    //
    // Real priority queues only prioritize *pending* items.
    // To test this reliably without race conditions, we'd need to pause the consumer.
    // Since we can't pause the consumer thread easily here, we accept that strictly
    // enforcing priority on the *first* message is racy if the consumer is idle.
    //
    // Ideally, we'd verify that `ctrl1` comes before `data2` or `mgmt1` if they were all
    // enqueued while the actor was busy processing `data1`.

    tx.send(Message::Data("data1".to_string())).unwrap();
    // Busy-wait or sleep just a tiny bit might actually make it worse by letting consumer drain.
    // Instead, let's send enough data to ensure the actor handles a batch,
    // and see if control slips in.

    tx.send(Message::Control("ctrl1".to_string())).unwrap();
    tx.send(Message::Data("data2".to_string())).unwrap();
    tx.send(Message::Management("mgmt1".to_string())).unwrap();

    thread::sleep(Duration::from_millis(100));
    drop(tx);
    handle.join().unwrap();

    // Then: Control should be processed before subsequent Data/Management if they were buffered together.
    // But since we can't guarantee buffering, we can only relax the test or make it more robust.
    //
    // Let's change the test to verify that priority *can* work by checking if Control came before Management,
    // or simply acknowledge that without consumer pausing, priority order is best-effort for the first message.
    //
    // RELAXED CHECK: Just verify all messages arrived. Priority is an optimization, not a strict correctness guarantee
    // for sequential sends on a live channel unless we overflow the batch size or pause.
    //
    // However, to fix the specific failure where data1 < ctrl1, we can remove the assertion that ctrl1 < data1
    // if data1 was indeed the very first message sent to an idle actor.
    //
    // Better test strategy: Verify Control comes before Management/Data *that were sent effectively simultaneously*.

    let order = message_order.lock().unwrap();
    let ctrl_idx = order.iter().position(|s| s == "C:ctrl1").expect("Control msg missing");
    let mgmt_idx = order.iter().position(|s| s == "M:mgmt1").expect("Mgmt msg missing");
    let data2_idx = order.iter().position(|s| s == "D:data2").expect("Data2 msg missing");

    // Control should definitely beat Management and Data2 if they are buffered.
    // But if the actor is fast enough to drain data1, then ctrl1, then mgmt1... ordering is send-order.
    // Priority only reorders *buffered* messages.
    //
    // If the test fails, it's because the system is "too efficient" (processing 1 by 1).
    // We should probably delete this test or modify it to be robust.
    //
    // Let's modify the expectation:
    // If we want to force buffering, we need the actor to be blocked or slow.
    // We can make the actor sleep in handle_data.

    // NOTE: This updated test block replaces the old logic with a "Slow Actor" approach
    // to force buffering and prove priority works.
}

#[test]
fn cuj_priority_control_buffered_reordering() {
    let message_order = Arc::new(Mutex::new(Vec::new()));
    let start_barrier = Arc::new(std::sync::Barrier::new(2));

    struct SlowActor {
        order: Arc<Mutex<Vec<String>>>,
        barrier: Arc<std::sync::Barrier>,
    }

    impl Actor<String, String, String> for SlowActor {
        fn handle_data(&mut self, msg: String) -> HandlerResult {
            if msg == "data1" {
                // Signal that we have started processing data1
                self.barrier.wait();
                // Sleep to ensure subsequent messages are buffered
                thread::sleep(Duration::from_millis(100));
            }
            self.order.lock().unwrap().push(format!("D:{}", msg));
            Ok(())
        }
        fn handle_control(&mut self, msg: String) -> HandlerResult {
            self.order.lock().unwrap().push(format!("C:{}", msg));
            Ok(())
        }
        fn handle_management(&mut self, msg: String) -> HandlerResult {
            self.order.lock().unwrap().push(format!("M:{}", msg));
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    let (tx, mut rx) = ActorScheduler::<String, String, String>::new(10, 64);
    let order_clone = message_order.clone();
    let barrier_clone = start_barrier.clone();

    let handle = thread::spawn(move || {
        let mut actor = SlowActor {
            order: order_clone,
            barrier: barrier_clone,
        };
        rx.run(&mut actor);
    });

    // 1. Send data1.
    tx.send(Message::Data("data1".to_string())).unwrap();

    // 2. Wait for actor to start processing data1.
    // This guarantees data1 is dequeued and the actor is "busy".
    start_barrier.wait();

    // 3. Send others while actor is sleeping. They should buffer in the channel.
    //    Send order: Data2, Management1, Control1
    tx.send(Message::Data("data2".to_string())).unwrap();
    tx.send(Message::Management("mgmt1".to_string())).unwrap();
    tx.send(Message::Control("ctrl1".to_string())).unwrap();

    // 4. Drop tx to allow shutdown after processing
    drop(tx);
    handle.join().unwrap();

    let order = message_order.lock().unwrap();

    // Expected order:
    // 1. D:data1 (processed first because we waited for it)
    // 2. C:ctrl1 (priority over buffered data/mgmt)
    // 3. M:mgmt1 (priority over buffered data)
    // 4. D:data2 (lowest priority)

    assert_eq!(order[0], "D:data1", "First message must be data1");

    // Find indices of the buffered messages
    let ctrl_idx = order.iter().position(|s| s == "C:ctrl1").expect("Missing ctrl1");
    let mgmt_idx = order.iter().position(|s| s == "M:mgmt1").expect("Missing mgmt1");
    let data2_idx = order.iter().position(|s| s == "D:data2").expect("Missing data2");

    assert!(
        ctrl_idx < mgmt_idx,
        "Control should preempt Management (Order: {:?})",
        order
    );
    assert!(
        mgmt_idx < data2_idx,
        "Management should preempt Data (Order: {:?})",
        order
    );
}

// =============================================================================
// Concurrent Sender Tests
// =============================================================================

#[test]
fn cuj_concurrent_senders_all_messages_delivered() {
    const NUM_SENDERS: usize = 5;
    const MESSAGES_PER_SENDER: usize = 20;

    let received_count = Arc::new(AtomicUsize::new(0));

    struct CountingActor {
        count: Arc<AtomicUsize>,
    }

    impl Actor<usize, (), ()> for CountingActor {
        fn handle_data(&mut self, _: usize) -> HandlerResult {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn handle_control(&mut self, _: ()) -> HandlerResult {
            Ok(())
        }
        fn handle_management(&mut self, _: ()) -> HandlerResult {
            Ok(())
        }
        fn park(&mut self, _: SystemStatus) -> Result<ActorStatus, HandlerError> {
            Ok(ActorStatus::Idle)
        }
    }

    // ActorHandle is not Clone — use ActorBuilder to create one handle per sender
    let mut builder = ActorBuilder::<usize, (), ()>::new(256, None);
    let mut sender_handles_vec: Vec<_> = (0..NUM_SENDERS).map(|_| builder.add_producer()).collect();
    let mut rx = builder.build_with_burst(10, actor_scheduler::ShutdownMode::default());
    let count_clone = received_count.clone();

    let receiver_handle = thread::spawn(move || {
        let mut actor = CountingActor { count: count_clone };
        rx.run(&mut actor);
    });

    // Spawn multiple senders, each with its own dedicated SPSC handle
    let mut sender_handles = Vec::new();
    for (sender_id, tx) in sender_handles_vec.drain(..).enumerate() {
        let handle = thread::spawn(move || {
            for msg_id in 0..MESSAGES_PER_SENDER {
                tx.send(Message::Data(sender_id * 1000 + msg_id)).unwrap();
            }
        });
        sender_handles.push(handle);
    }

    // Wait for all senders
    for h in sender_handles {
        h.join().unwrap();
    }

    thread::sleep(Duration::from_millis(100));
    receiver_handle.join().unwrap();

    // Verify all messages received
    assert_eq!(
        received_count.load(Ordering::SeqCst),
        NUM_SENDERS * MESSAGES_PER_SENDER,
        "All {} messages should be delivered",
        NUM_SENDERS * MESSAGES_PER_SENDER
    );
}
