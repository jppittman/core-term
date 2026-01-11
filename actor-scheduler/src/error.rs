//! Error types for the actor scheduler library.

use std::sync::mpsc;
use std::fmt;

/// Error returned when sending to an actor fails.
///
/// This can occur when:
/// - The receiving actor has been dropped (channel disconnected)
/// - Retry attempts have been exhausted (for Control/Management lanes)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]

pub enum SendError {
    Timeout,
    Unknown,
}

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SendError::Unknown => write!(f, "failed to send message to actor"),
            SendError::Timeout => write!(f, "timeout sending message to actor"),
        }
    }
}

impl std::error::Error for SendError {}

// Map std::sync::mpsc errors to our opaque SendError
impl<T> From<mpsc::SendError<T>> for SendError {
    fn from(_: mpsc::SendError<T>) -> Self {
        SendError::Unknown
    }
}

impl<T> From<mpsc::TrySendError<T>> for SendError {
    fn from(_: mpsc::TrySendError<T>) -> Self {
        SendError::Unknown
    }
}

/// Error returned by actor message handlers.
///
/// Errors are classified into two categories for supervision purposes:
/// - **Retriable**: Transient errors that may succeed on retry (e.g., temporary resource exhaustion).
///   A supervisor would restart the actor and allow it to continue processing.
/// - **Fatal**: Permanent errors that indicate a bug or unrecoverable state (e.g., invariant violation).
///   A supervisor would crash the entire troupe to prevent cascading failures.
///
/// # Examples
///
/// ```ignore
/// // Retriable error - actor can be restarted
/// return Err(ActorError::retriable("temporary file system error"));
///
/// // Fatal error - troupe should crash
/// return Err(ActorError::fatal("invariant violated: expected positive value"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActorError {
    message: String,
    retriable: bool,
}

impl ActorError {
    /// Create a retriable error.
    ///
    /// Indicates a transient failure that a supervisor could recover from by restarting the actor.
    pub fn retriable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retriable: true,
        }
    }

    /// Create a fatal error.
    ///
    /// Indicates a permanent failure that should crash the entire troupe.
    pub fn fatal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retriable: false,
        }
    }

    /// Returns true if this error is retriable (actor can be restarted).
    pub fn is_retriable(&self) -> bool {
        self.retriable
    }

    /// Returns true if this error is fatal (should crash the troupe).
    pub fn is_fatal(&self) -> bool {
        !self.retriable
    }

    /// Get the error message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ActorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let severity = if self.retriable { "retriable" } else { "fatal" };
        write!(f, "{} actor error: {}", severity, self.message)
    }
}

impl std::error::Error for ActorError {}
