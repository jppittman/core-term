// src/platform/mod.rs
//
// Platform module. Most logic moved to pixelflow-engine.

pub mod actions;
pub use pixelflow_engine::platform::waker;

// Re-exports
// pub use pixelflow_engine::platform::EnginePlatform; // Accessed via pixelflow_engine directly in main.rs
