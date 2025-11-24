use crate::platform::waker::{CocoaWaker, EventLoopWaker};
use crate::platform::GenericPlatform;
use crate::renderer::RenderChannels;
use anyhow::Result;
use log::*;

/// macOS platform implementation - thin wrapper around GenericPlatform.
/// All the actual logic is in GenericPlatform; this just provides the macOS-specific type.
pub struct MacosPlatform {
    inner: GenericPlatform,
}

impl MacosPlatform {
    pub fn new(
        channels: crate::platform::PlatformChannels,
        render_channels: RenderChannels,
    ) -> Result<Self> {
        info!("MacosPlatform::new() - Delegating to GenericPlatform");
        let inner = GenericPlatform::new(channels, render_channels)?;
        Ok(Self { inner })
    }

    /// Create a waker for signaling the event loop from background threads.
    pub fn create_waker(&self) -> Result<Box<dyn EventLoopWaker>> {
        Ok(Box::new(CocoaWaker::new()))
    }

    pub fn run(self) -> Result<()> {
        info!("MacosPlatform::run() - Delegating to GenericPlatform");
        self.inner.run()
    }
}
