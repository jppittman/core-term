//! Why a scheduler stopped.

/// The reason an actor's scheduler exited.
///
/// Returned by [`ActorScheduler::run`](crate::ActorScheduler::run) and carried by
/// [`Step::Halted`](crate::mealy::Step::Halted), so a caller can tell a clean stop from a
/// failure and decide whether to restart.
///
/// Both variants are actually produced. There is deliberately no `Pending`/`Running`/
/// `Terminating` here: a phase nobody returns is a phase nobody can test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Exit {
    /// Clean stop: `Message::Shutdown` received, or all sender handles dropped.
    Completed,

    /// Stopped by `HandlerError::Recoverable`. The message describes the failure.
    ///
    /// `HandlerError::Fatal` is never represented here — it panics immediately.
    Failed(String),
}

impl Exit {
    /// Returns `true` if this is an abnormal exit.
    #[must_use]
    pub fn is_failed(&self) -> bool {
        matches!(self, Exit::Failed(_))
    }

    /// Returns `true` if this is a clean exit.
    #[must_use]
    pub fn is_completed(&self) -> bool {
        matches!(self, Exit::Completed)
    }
}

impl std::fmt::Display for Exit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Exit::Completed => write!(f, "Completed"),
            Exit::Failed(msg) => write!(f, "Failed({msg})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_helpers_distinguish_the_two_outcomes() {
        assert!(Exit::Failed("x".into()).is_failed());
        assert!(!Exit::Completed.is_failed());
        assert!(Exit::Completed.is_completed());
        assert!(!Exit::Failed("x".into()).is_completed());
    }

    #[test]
    fn exit_display_formats_each_variant() {
        assert_eq!(Exit::Completed.to_string(), "Completed");
        assert_eq!(Exit::Failed("boom".into()).to_string(), "Failed(boom)");
    }
}
