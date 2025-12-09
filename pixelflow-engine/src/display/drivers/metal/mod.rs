//! Metal display driver module.
//!
//! This module contains:
//! - wrappers.rs: Type-safe Rust wrappers around Metal FFI
//! - driver.rs: The actual display driver implementation

pub mod wrappers;
mod driver;

pub use driver::MetalDisplayDriver;
