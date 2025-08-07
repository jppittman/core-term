// src/platform/linux_x11/tests.rs

use super::*;
use crate::platform::backends::mock::MockDriver;

#[test]
fn it_should_create_a_new_linux_x11_platform() {
    let platform = LinuxX11Platform::<MockDriver>::new(
        80,
        24,
        "sh".to_string(),
        Vec::new(),
    );
    assert!(platform.is_ok());
}
