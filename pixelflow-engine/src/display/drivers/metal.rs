//! Metal Display Driver (Type Alias)
//!
//! This module now just defines the type alias for the Generic Driver
//! specialized for macOS.

use crate::display::generic::Driver;
use crate::platform::macos::application::MacApplication;
use crate::platform::macos::window::MacWindow;

pub type MetalDisplayDriver = Driver<MacApplication>;
