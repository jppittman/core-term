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

impl<T> From<crate::spsc::TrySendError<T>> for SendError {
    fn from(err: crate::spsc::TrySendError<T>) -> Self {
        match err {
            crate::spsc::TrySendError::Full(_) => SendError::Timeout,
            crate::spsc::TrySendError::Disconnected(_) => SendError::Disconnected,
        }
    }
}

/// Error from an actor handler indicating failure severity.
///
/// Generic over `E`, the error payload type. The scheduler converts `E` to
/// `String` (via `Display`) at the `PodPhase` boundary, so actors can use
/// domain-specific error types internally while the supervisor sees strings.
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
///
/// # Default type parameter
///
/// `E` defaults to `String` for backward compatibility. Existing code using
/// `HandlerError` (without a type parameter) continues to work unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandlerError<E = String> {
    /// Actor needs restart - scheduler exits, supervisor may respawn
    Recoverable(E),
    /// Process must crash - scheduler panics
    Fatal(E),
}

impl<E> HandlerError<E> {
    /// Create a recoverable error (actor restart).
    pub fn recoverable(e: impl Into<E>) -> Self {
        HandlerError::Recoverable(e.into())
    }

    /// Create a fatal error (process crash).
    pub fn fatal(e: impl Into<E>) -> Self {
        HandlerError::Fatal(e.into())
    }
}

impl<E: std::fmt::Display> std::fmt::Display for HandlerError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandlerError::Recoverable(e) => write!(f, "recoverable: {}", e),
            HandlerError::Fatal(e) => write!(f, "FATAL: {}", e),
        }
    }
}

impl<E: std::fmt::Display + std::fmt::Debug> std::error::Error for HandlerError<E> {}

/// Result type for actor handlers.
///
/// `E` defaults to `String` for backward compatibility. Actors with custom
/// error types use `HandlerResult<MyError>`.
pub type HandlerResult<E = String> = Result<(), HandlerError<E>>;

/// Result of draining messages from a channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainStatus {
    /// Channel empty, all messages processed
    Empty,
    /// Hit burst limit, more messages may be available
    More,
    /// Channel disconnected (senders dropped)
    Disconnected,
}
