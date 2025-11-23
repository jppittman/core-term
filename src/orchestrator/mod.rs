// src/orchestrator/mod.rs

pub mod orchestrator_actor;
pub mod orchestrator_channel;

pub use orchestrator_channel::{OrchestratorEvent, OrchestratorSender};

#[cfg(test)]
mod tests;
