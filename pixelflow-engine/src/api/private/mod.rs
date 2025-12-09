//! Internal engine message types.
//!
//! These types are used for communication between engine components
//! (driver, scheduler, vsync actor, etc.) and are not part of the public API.

// All internal message types consolidated in api/messages.rs
pub use crate::api::messages::*;

// Re-export vsync types for internal use
pub use crate::vsync_actor::{VsyncActorHandle, VsyncRequest, VsyncMessage};
