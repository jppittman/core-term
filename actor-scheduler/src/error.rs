//! Error types for the actor scheduler library.

use std::sync::mpsc;

/// Error returned when sending to an actor fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendError {
    /// Timed out waiting for channel capacity
    Timeout,
    /// Channel disconnected - receiver dropped
    Disconnected,
}

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SendError::Timeout => write!(f, "timeout sending message to actor"),
            SendError::Disconnected => write!(f, "actor channel disconnected"),
        }
    }
}

impl std::error::Error for SendError {}

impl<T> From<mpsc::SendError<T>> for SendError {
    fn from(_: mpsc::SendError<T>) -> Self {
        SendError::Disconnected
    }
}

impl<T> From<mpsc::TrySendError<T>> for SendError {
    fn from(err: mpsc::TrySendError<T>) -> Self {
        match err {
            mpsc::TrySendError::Full(_) => SendError::Timeout,
            mpsc::TrySendError::Disconnected(_) => SendError::Disconnected,
        }
    }
}

/// Error from an actor handler indicating failure severity.
///
/// # Severity Levels
///
/// - **Recoverable**: Actor state is corrupted but the problem might be fixed
///   by restarting. The scheduler exits `run()` and the supervisor can respawn.
///
/// - **Fatal**: Unrecoverable process-level failure. The scheduler panics,
///   bringing down the entire process. Use for invariant violations, data
///   corruption, or conditions where continuing would cause harm.
///
/// # When to use each
///
/// Return `Recoverable` when:
/// - Actor state became inconsistent
/// - A resource leaked or became unavailable
/// - Retries exhausted on a critical operation
///
/// Return `Fatal` when:
/// - Memory corruption detected
/// - Security invariant violated
/// - Continuing would cause data loss or undefined behavior
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandlerError {
    /// Actor needs restart - scheduler exits, supervisor may respawn
    Recoverable(String),
    /// Process must crash - scheduler panics
    Fatal(String),
}

impl HandlerError {
    /// Create a recoverable error (actor restart).
    pub fn recoverable(msg: impl Into<String>) -> Self {
        HandlerError::Recoverable(msg.into())
    }

    /// Create a fatal error (process crash).
    pub fn fatal(msg: impl Into<String>) -> Self {
        HandlerError::Fatal(msg.into())
    }
}

impl std::fmt::Display for HandlerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandlerError::Recoverable(msg) => write!(f, "recoverable: {}", msg),
            HandlerError::Fatal(msg) => write!(f, "FATAL: {}", msg),
        }
    }
}

impl std::error::Error for HandlerError {}

/// Result type for actor handlers.
pub type HandlerResult = Result<(), HandlerError>;

/// Result of draining messages from a channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DrainStatus {
    /// Channel empty, all messages processed
    Empty,
    /// Hit burst limit, more messages may be available
    More,
    /// Channel disconnected (senders dropped)
    Disconnected,
}
