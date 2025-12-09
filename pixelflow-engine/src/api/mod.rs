//! API types for the pixelflow engine.
//!
//! This module is split into:
//! - `public`: Types exposed to applications (EngineEvent, AppAction, Application trait)
//! - `private`: Internal engine types (EngineControl, DisplayEvent, DriverCommand)

pub mod public;
pub mod private;
pub mod messages;

// Re-export public API at crate root for convenience
pub use public::*;
