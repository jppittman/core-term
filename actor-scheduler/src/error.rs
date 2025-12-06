//! Error types for the actor scheduler library.

use std::sync::mpsc;

/// Error returned when sending to an actor fails.
///
/// This can occur when:
/// - The receiving actor has been dropped (channel disconnected)
/// - Retry attempts have been exhausted (for Control/Management lanes)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SendError;

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "failed to send message to actor")
    }
}

impl std::error::Error for SendError {}

// Map std::sync::mpsc errors to our opaque SendError
impl<T> From<mpsc::SendError<T>> for SendError {
    fn from(_: mpsc::SendError<T>) -> Self {
        SendError
    }
}

impl<T> From<mpsc::TrySendError<T>> for SendError {
    fn from(_: mpsc::TrySendError<T>) -> Self {
        SendError
    }
}
