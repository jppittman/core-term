//! # PTY I/O and Event Monitoring
//!
//! Manages pseudo-terminal (PTY) communication and multiplexed reading/parsing/writing.
//!
//! ## Architecture: Three-Thread Pipeline
//!
//! ```text
//! PTY (File Descriptor)
//!      ↓ (nonblocking reads via kqueue/epoll)
//! ┌─────────────────────────────────────────────────┐
//! │ Read Thread (read_thread)                       │
//! │ - Polls PTY with kqueue/epoll (efficient!)      │
//! │ - Reads raw bytes when available                │
//! │ - Sends bytes to Parser via ActorScheduler      │
//! │ - Recycles buffers from parser                  │
//! └─────────────────────────────────────────────────┘
//!      │ (Vec<u8> batches)
//!      │
//! ┌─────────────────────────────────────────────────┐
//! │ Parser Thread (parser_thread)                   │
//! │ - Receives raw bytes from ActorScheduler        │
//! │ - Parses ANSI commands (CPU-intensive!)         │
//! │ - Sends AnsiCommand to app via SyncSender       │
//! │ - Returns buffers to read thread for recycling  │
//! └─────────────────────────────────────────────────┘
//!      │ (Vec<AnsiCommand>)
//!      │
//!      → App/Terminal State Machine
//!
//! PTY (File Descriptor)
//!      ↑ (nonblocking writes)
//! ┌─────────────────────────────────────────────────┐
//! │ Write Thread (write_thread)                     │
//! │ - Receives write requests from app              │
//! │ - Handles PTY resize commands (TIOCSWINSZ)      │
//! │ - Owns PTY lifecycle (RAII)                     │
//! └─────────────────────────────────────────────────┘
//!      ↑ (from app/shell commands)
//! ```
//!
//! ## Design Rationale: Three Separate Threads
//!
//! Why not a single thread or two threads?
//!
//! **Single thread**: Sequential read → parse → write. Parsing (CPU) blocks I/O threads
//! (epoll/kqueue can't wake while parsing). Terminal feels sluggish.
//!
//! **Two threads**: Read + Parse in one thread; write separate. Parsing delays reading;
//! app output causes lag while parser catches up.
//!
//! **Three threads** (this design): Maximizes parallelism:
//! - Read thread is **always ready** to accept data from PTY
//! - Parser thread can take its time without blocking reads
//! - Write thread can send output while read/parse are busy
//! - CPU, I/O, and app latency are decoupled
//!
//! ## Thread Communication
//!
//! ### Read → Parser: ActorScheduler
//! - **Type**: Burst-limited actor channel
//! - **Data**: Vec<u8> (raw bytes from PTY)
//! - **Purpose**: Backpressure control; prevents read thread from flooding parser
//! - **Lifetime**: Closed when read thread stops (PTY closed)
//!
//! ### Parser → App: SyncSender
//! - **Type**: Synchronous channel (blocking send)
//! - **Data**: Vec<AnsiCommand> (parsed ANSI commands)
//! - **Purpose**: Delivers parsed commands to terminal state machine
//! - **Blocking**: Parser blocks if app can't keep up (natural backpressure)
//!
//! ### App → Write: Receiver
//! - **Type**: Synchronous channel (blocking recv)
//! - **Data**: Vec<u8> (bytes to write to PTY)
//! - **Purpose**: App sends shell input/output to PTY
//!
//! ### Parser → Read: Buffer Recycling
//! - **Type**: MPMC channel (unbounded)
//! - **Data**: Vec<u8> (empty buffers)
//! - **Purpose**: Read thread reuses buffers instead of allocating each time
//! - **Safe**: Closed loop; buffer count is bounded
//!
//! ## Platform-Specific Event Notification
//!
//! ### macOS: kqueue
//! The read thread uses **kqueue** for efficient PTY monitoring:
//! ```text
//! kqueue_fd = kqueue()
//! kevent(register EVFILT_READ on PTY master FD)
//! kevent(poll) → returns when PTY has data
//! read(pty_fd, buffer, size)
//! ```
//!
//! Advantages:
//! - Sub-millisecond latency (instant wakeup when data arrives)
//! - **Edge-triggered**: Fires only when state changes
//! - CPU-efficient (no polling loop)
//!
//! ### Linux: epoll
//! Uses **epoll** (equivalent to kqueue on Linux):
//! ```text
//! epoll_fd = epoll_create()
//! epoll_ctl(add, pty_fd, EPOLLIN)
//! epoll_wait() → returns when PTY has data
//! read(pty_fd, buffer, size)
//! ```
//!
//! Advantages: Same as kqueue, Linux-native
//!
//! ## PTY Management
//!
//! Handled by `NixPty` (pty.rs):
//!
//! 1. **Creation**: `openpty()` creates a PTY pair (master, slave)
//! 2. **Forking**: Parent keeps master FD, forks child process
//! 3. **Child setup**: Child becomes session leader, connects to slave PTY
//! 4. **Execution**: Child execs shell (bash, zsh, etc.)
//! 5. **Resizing**: Parent sends `TIOCSWINSZ` ioctl to resize PTY window
//! 6. **Cleanup**: Dropping `NixPty` closes master FD; child PTY is closed automatically
//!
//! Lifecycle ownership:
//! - Read thread has a **clone** of the master FD (can be closed independently)
//! - Write thread owns the **primary** FD (drops last → closes PTY)
//! - When write thread drops, read thread's events stop firing (FD is invalid)
//!
//! ## Buffer Recycling: Zero-Allocation After Startup
//!
//! The read thread pre-allocates buffers and recycles them:
//!
//! ```text
//! Startup:
//!   read_thread allocates N buffers and sends to parser
//!
//! Runtime (per cycle):
//!   1. parser returns empty buffer via recycler_tx
//!   2. read_thread reuses buffer (no new allocation)
//!   3. read_thread fills buffer and sends to parser
//!   Repeat
//! ```
//!
//! **Benefit**: Zero garbage collection pauses after startup
//!
//! ## Backpressure and Flow Control
//!
//! The ActorScheduler (read → parser) enforces backpressure:
//!
//! ```text
//! Normal state:
//!   read_thread sends byte batches → parser_thread consumes
//!
//! If parser is slow (e.g., large ANSI sequence):
//!   read_thread sends batch → ActorScheduler buffer fills
//!   read_thread blocks on ActorScheduler send
//!   (PTY events are buffered by OS, not lost)
//!
//! Once parser catches up:
//!   read_thread resumes reading
//! ```
//!
//! This prevents memory blow-up if parsing can't keep pace with PTY data.
//!
//! ## Error Handling and Cleanup
//!
//! If any thread panics or errors:
//! - **Read thread fails**: Parser thread will stop when ActorScheduler closes
//! - **Parser thread fails**: App won't receive commands; write thread continues
//! - **Write thread fails**: App loses ability to send input; read thread continues receiving
//! - **Any Drop**: EventMonitorActor::drop() closes write thread first (kills PTY), then read, then parser
//!
//! ## Performance Characteristics
//!
//! | Operation | Latency | Throughput |
//! |-----------|---------|-----------|
//! | PTY read (kqueue) | <1 ms | ~1 MB/s (raw I/O) |
//! | ANSI parsing | ~5-20 ns/byte | ~50-100 MB/s (CPU-bound) |
//! | Total pipeline | ~1-5 ms | Limited by slower thread |
//!
//! For typical terminal (80 cols × 24 rows, 30 FPS):
//! - ~2000 bytes/frame
//! - ~100-400 microseconds parsing
//! - **Negligible** compared to rendering (~10 ms per frame)
//!
//! ## Testing and Debugging
//!
//! Enable logging for detailed I/O tracing:
//! ```bash
//! RUST_LOG=core_term::io=trace cargo run
//! ```
//!
//! This will log:
//! - Thread spawning/cleanup
//! - Read/write/parse events
//! - Buffer allocation/recycling
//! - Backpressure situations

pub mod event_monitor_actor;
pub mod pty;
pub mod traits;

/// Commands that can be sent to the PTY write thread.
///
/// The write thread handles both data writes and control operations like resize.
/// This unified command type allows the terminal app to communicate all PTY operations
/// through a single channel.
///
/// # Examples
///
/// ```ignore
/// // Send user input to the shell
/// pty_tx.send(PtyCommand::Write(b"ls -la\n".to_vec()))?;
///
/// // Resize the PTY when window size changes
/// pty_tx.send(PtyCommand::Resize { cols: 120, rows: 40 })?;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtyCommand {
    /// Write raw bytes to the PTY.
    ///
    /// These bytes are sent directly to the shell process via the PTY master FD.
    Write(Vec<u8>),

    /// Resize the PTY window.
    ///
    /// This triggers:
    /// 1. `ioctl(TIOCSWINSZ)` to set the new window size
    /// 2. `SIGWINCH` signal to notify the shell of the size change
    ///
    /// Full-screen programs (vim, less, top, etc.) will respond to SIGWINCH
    /// by querying the new size and redrawing.
    Resize {
        /// Number of columns in the terminal grid.
        cols: u16,
        /// Number of rows in the terminal grid.
        rows: u16,
    },
}

#[cfg(test)]
mod pty_tests;

// Platform-specific event monitoring implementations
#[cfg(target_os = "macos")]
pub mod kqueue;

#[cfg(target_os = "linux")]
pub mod epoll;

// Platform-agnostic re-exports
#[cfg(target_os = "macos")]
pub mod event {
    pub use super::kqueue::*;
}

#[cfg(target_os = "linux")]
pub mod event {
    pub use super::epoll::*;
}
