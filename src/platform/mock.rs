// src/platform/mock.rs

use crate::platform::actions::PlatformAction;
use crate::platform::backends::{BackendEvent, PlatformState};
use crate::platform::platform_trait::Platform;
use crate::platform::PlatformEvent;
use anyhow::Result;

pub struct MockPlatform {
    events: Vec<PlatformEvent>,
    dispatched_actions: Vec<PlatformAction>,
}

impl MockPlatform {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            dispatched_actions: Vec::new(),
        }
    }

    pub fn push_event(&mut self, event: PlatformEvent) {
        self.events.push(event);
    }

    pub fn dispatched_actions(&self) -> &[PlatformAction] {
        &self.dispatched_actions
    }
}

impl Platform for MockPlatform {
    fn new(
        _initial_pty_cols: u16,
        _initial_pty_rows: u16,
        _shell_command: String,
        _shell_args: Vec<String>,
    ) -> Result<(Self, PlatformState)> {
        Ok((
            Self::new(),
            PlatformState {
                event_fd: None,
                font_cell_width_px: 8,
                font_cell_height_px: 16,
                scale_factor: 1.0,
                display_width_px: 800,
                display_height_px: 600,
            },
        ))
    }

    fn poll_events(&mut self) -> Result<Vec<PlatformEvent>> {
        Ok(self.events.drain(..).collect())
    }

    fn dispatch_actions(&mut self, actions: Vec<PlatformAction>) -> Result<()> {
        self.dispatched_actions.extend(actions);
        Ok(())
    }

    fn get_current_platform_state(&self) -> PlatformState {
        PlatformState {
            event_fd: None,
            font_cell_width_px: 8,
            font_cell_height_px: 16,
            scale_factor: 1.0,
            display_width_px: 800,
            display_height_px: 600,
        }
    }

    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}
