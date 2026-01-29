//! Metal Display Driver (Type Alias)
//!
//! This module defines the type alias for the macOS display driver
//! using the new PlatformActor architecture.

use crate::display::driver::DriverActor;
use crate::display::platform::PlatformActor;
use crate::platform::macos::platform::MetalOps;

/// The macOS display driver: DriverActor wrapping PlatformActor<MetalOps>.
pub type MetalDisplayDriver = DriverActor<PlatformActor<MetalOps>>;
