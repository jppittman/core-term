// src/platform/backends/mock.rs

use crate::platform::backends::{
    BackendEvent, CursorVisibility, Driver, FocusState, PlatformState, RenderCommand,
};
use anyhow::Result;
use std::os::unix::io::RawFd;

pub struct MockDriver {
    events: Vec<BackendEvent>,
    render_commands: Vec<RenderCommand>,
    framebuffer: Vec<u8>,
}

impl MockDriver {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            render_commands: Vec::new(),
            framebuffer: vec![0u8; 800 * 600 * 4], // Default 800x600 RGBA
        }
    }

    pub fn push_event(&mut self, event: BackendEvent) {
        self.events.push(event);
    }

    pub fn render_commands(&self) -> &[RenderCommand] {
        &self.render_commands
    }
}

impl Driver for MockDriver {
    fn new() -> Result<Self> {
        Ok(Self::new())
    }

    fn get_event_fd(&self) -> Option<RawFd> {
        None
    }

    fn process_events(&mut self) -> Result<Vec<BackendEvent>> {
        Ok(self.events.drain(..).collect())
    }

    fn get_platform_state(&self) -> PlatformState {
        PlatformState {
            event_fd: None,
            font_cell_width_px: 8,
            font_cell_height_px: 16,
            scale_factor: 1.0,
            display_width_px: 800,
            display_height_px: 600,
        }
    }

    fn execute_render_commands(&mut self, commands: Vec<RenderCommand>) -> Result<()> {
        self.render_commands.extend(commands);
        Ok(())
    }

    fn present(&mut self) -> Result<()> {
        Ok(())
    }

    fn set_title(&mut self, _title: &str) {}

    fn bell(&mut self) {}

    fn set_cursor_visibility(&mut self, _visibility: CursorVisibility) {}

    fn set_focus(&mut self, _focus_state: FocusState) {}

    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }

    fn get_framebuffer_mut(&mut self) -> &mut [u8] {
        &mut self.framebuffer
    }

    fn get_framebuffer_size(&self) -> (usize, usize) {
        (800, 600)
    }
}
