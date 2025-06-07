// src/platform/mod.rs

//! This module abstracts platform-specific functionalities, including UI backend interactions,
//! PTY management, and OS-level event handling. It aims to provide a consistent interface
//! for the `AppOrchestrator` to interact with different underlying platforms like Linux/X11,
//! Linux/Wayland (planned), or others.
//!
//! Key components:
//! - `Platform` trait: Defines the common interface for all platform implementations.
//! - Concrete platform structs (e.g., `LinuxX11Platform`, `WaylandPlatform`): Implement the `Platform` trait.
//! - `backends` submodule: Contains `Driver` implementations for specific UI toolkits or rendering methods (X11, console, Wayland).
//! - `os` submodule: Provides OS-specific utilities like PTY management (`NixPty`) and event monitoring (`Epoll`).
//! - `actions` and `events`: Define standardized commands and event types for communication between the orchestrator and the platform.

// Publicly export key types and traits for use by other modules (e.g., main.rs, orchestrator.rs).
pub mod actions;
pub mod backends; // Exposes Driver trait, BackendEvent, RenderCommand, etc.
pub mod os;
pub mod platform_trait; // Exposes the Platform trait

// Concrete Platform Implementations
// Conditionally compile and export platform-specific modules and types.

// Linux X11 Platform (default for Linux if Wayland feature is not enabled)
#[cfg(all(target_os = "linux", not(feature = "wayland")))]
pub mod linux_x11;
#[cfg(all(target_os = "linux", not(feature = "wayland")))]
pub use linux_x11::LinuxX11Platform;

// Linux Wayland Platform (enabled with "wayland" feature)
#[cfg(all(target_os = "linux", feature = "wayland"))]
pub mod linux_wayland;
#[cfg(all(target_os = "linux", feature = "wayland"))]
pub use linux_wayland::WaylandPlatform;

// Console Platform (Example - could be enabled via a feature or specific build target)
// pub mod console_platform;
// pub use console_platform::ConsolePlatform;

// TODO: Add WASM platform when implemented
// #[cfg(target_arch = "wasm32")]
// pub mod wasm_platform;
// #[cfg(target_arch = "wasm32")]
// pub use wasm_platform::WasmPlatform;

/// Represents events that the `AppOrchestrator` receives from the `Platform`.
/// This enum consolidates different types of events that can originate from
/// the platform layer, such as UI events (`BackendEvent`) or PTY I/O data.
#[derive(Debug, Clone)]
pub enum PlatformEvent {
    /// An event originating from the UI backend (e.g., keyboard input, resize).
    Backend(backends::BackendEvent),
    /// Data received from the PTY.
    IOEvent { data: Vec<u8> },
    // Other platform-level events can be added here if needed,
    // e.g., PTYClosed, PlatformError, etc.
}

// Implement From<BackendEvent> for PlatformEvent for convenience.
impl From<backends::BackendEvent> for PlatformEvent {
    fn from(event: backends::BackendEvent) -> Self {
        PlatformEvent::Backend(event)
    }
}
