//! Ready-to-go actors.
//!
//! Small, general-purpose actors that most systems end up needing and would otherwise
//! hand-roll once per crate. Nothing here knows anything about the application using it —
//! if a type in this module mentions a domain concept, it's in the wrong place.

pub mod timer;

pub use timer::{Schedule, Timer};
