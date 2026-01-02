//! Work Pool Actor Demo
//!
//! This example demonstrates using WorkPoolActor as a concrete alternative
//! to a full work-stealing scheduler inversion.
//!
//! ## Scenario
//!
//! A data processing pipeline with:
//! - 1 dedicated TerminalApp actor (latency-critical)
//! - 1 WorkPoolActor handling batch data processing (throughput-oriented)
//!
//! ## Key Insight
//!
//! The work pool IS the work-stealing model, just implemented as an actor
//! in the current system. No architectural changes needed!

use actor_scheduler::{Actor, ActorHandle, ActorScheduler, Message, ParkHint, WorkPoolActor, WorkPoolConfig};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// ============================================================================
// Example: Terminal Application (latency-critical, dedicated thread)
// ============================================================================

struct TerminalApp {
    name: String,
    keystrokes_processed: usize,
}

impl TerminalApp {
    fn new(name: String) -> Self {
        Self {
            name,
            keystrokes_processed: 0,
        }
    }
}

enum TerminalControl {
    Keystroke(char),
    Resize(u32, u32),
}

impl Actor<(), TerminalControl, ()> for TerminalApp {
    fn handle_control(&mut self, msg: TerminalControl) {
        match msg {
            TerminalControl::Keystroke(key) => {
                println!("[{}] Keystroke: {}", self.name, key);
                self.keystrokes_processed += 1;
            }
            TerminalControl::Resize(w, h) => {
                println!("[{}] Resize: {}x{}", self.name, w, h);
            }
        }
    }

    fn handle_management(&mut self, _msg: ()) {}
    fn handle_data(&mut self, _msg: ()) {}

    fn park(&mut self, hint: ParkHint) -> ParkHint {
        hint
    }
}

// ============================================================================
// Example: Data Processing Tasks (throughput-oriented, work pool)
// ============================================================================

#[derive(Debug, Clone)]
enum DataTask {
    Transform(String),
    Analyze(Vec<u8>),
    Aggregate(usize),
}

fn process_task(task: DataTask, counter: Arc<AtomicUsize>) {
    match task {
        DataTask::Transform(s) => {
            // Simulate CPU-intensive work
            thread::sleep(Duration::from_millis(5));
            println!("  [Worker] Transformed: {}", s);
            counter.fetch_add(1, Ordering::Relaxed);
        }
        DataTask::Analyze(data) => {
            thread::sleep(Duration::from_millis(10));
            println!("  [Worker] Analyzed {} bytes", data.len());
            counter.fetch_add(1, Ordering::Relaxed);
        }
        DataTask::Aggregate(count) => {
            thread::sleep(Duration::from_millis(2));
            println!("  [Worker] Aggregated {} items", count);
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// ============================================================================
// Main Demo
// ============================================================================

fn main() {
    println!("=== Work Pool Actor Demo ===\n");

    // Metrics
    let tasks_processed = Arc::new(AtomicUsize::new(0));
    let tasks_clone = Arc::clone(&tasks_processed);

    // -------------------------------------------------------------------------
    // 1. Create TerminalApp (dedicated thread, latency-critical)
    // -------------------------------------------------------------------------

    println!("1. Creating TerminalApp (dedicated thread)...");
    let (terminal_handle, mut terminal_scheduler) = ActorScheduler::new(10, 128);
    let mut terminal = TerminalApp::new("Terminal".to_string());

    let terminal_thread = thread::Builder::new()
        .name("terminal-app".to_string())
        .spawn(move || {
            terminal_scheduler.run(&mut terminal);
        })
        .unwrap();

    // -------------------------------------------------------------------------
    // 2. Create WorkPoolActor (4 workers, throughput-oriented)
    // -------------------------------------------------------------------------

    println!("2. Creating WorkPoolActor (4 workers)...\n");
    let config = WorkPoolConfig {
        num_workers: 4,
        thread_name_prefix: Some("data-worker".to_string()),
    };

    let mut work_pool = WorkPoolActor::new(config, move |task| {
        process_task(task, Arc::clone(&tasks_clone));
    });

    let (pool_handle, mut pool_scheduler): (ActorHandle<DataTask, DataTask, DataTask>, _) =
        ActorScheduler::new(1024, 128);

    let pool_thread = thread::Builder::new()
        .name("work-pool-supervisor".to_string())
        .spawn(move || {
            pool_scheduler.run(&mut work_pool);
        })
        .unwrap();

    // -------------------------------------------------------------------------
    // 3. Send mixed workload
    // -------------------------------------------------------------------------

    println!("3. Sending mixed workload...\n");

    // Send latency-critical keystrokes to Terminal (Control lane)
    for ch in "Hello".chars() {
        terminal_handle
            .send(Message::Control(TerminalControl::Keystroke(ch)))
            .unwrap();
        thread::sleep(Duration::from_millis(10)); // User typing speed
    }

    // Send throughput-oriented tasks to WorkPool (Data lane)
    println!("\n4. Sending batch tasks to work pool...\n");
    for i in 0..20 {
        let task = match i % 3 {
            0 => DataTask::Transform(format!("item-{}", i)),
            1 => DataTask::Analyze(vec![0u8; i * 100]),
            2 => DataTask::Aggregate(i),
            _ => unreachable!(),
        };
        pool_handle.send(Message::Data(task)).unwrap();
    }

    // Send high-priority task to work pool (Control lane - processed first)
    println!("5. Sending PRIORITY task to work pool...\n");
    pool_handle
        .send(Message::Control(DataTask::Transform("PRIORITY".to_string())))
        .unwrap();

    // -------------------------------------------------------------------------
    // 4. Wait for completion
    // -------------------------------------------------------------------------

    thread::sleep(Duration::from_millis(200));

    println!("\n=== Results ===");
    println!(
        "Tasks processed by work pool: {}",
        tasks_processed.load(Ordering::Relaxed)
    );

    // Shutdown
    drop(terminal_handle);
    drop(pool_handle);
    terminal_thread.join().unwrap();
    pool_thread.join().unwrap();

    println!("\n=== Key Observations ===");
    println!("✓ TerminalApp has dedicated thread (low latency)");
    println!("✓ WorkPoolActor has 4 workers (high throughput)");
    println!("✓ Both use same Actor trait (composable)");
    println!("✓ No changes to actor-scheduler core needed!");
    println!("\nThis IS the work-stealing model - just implemented as an actor.");
}
