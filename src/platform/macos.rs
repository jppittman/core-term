use crate::platform::backends::PlatformState;
use crate::platform::platform_trait::Platform;
use crate::platform::{GenericPlatform, PlatformEvent};
use anyhow::Result;
use log::*;

/// macOS platform implementation - thin wrapper around GenericPlatform.
/// All the actual logic is in GenericPlatform; this just provides the macOS-specific type.
pub struct MacosPlatform {
    inner: GenericPlatform,
}

impl Platform for MacosPlatform {
    fn new(channels: crate::platform::PlatformChannels) -> Result<Self>
    where
        Self: Sized,
    {
        info!("MacosPlatform::new() - Delegating to GenericPlatform");
        let inner = GenericPlatform::new(channels)?;
        Ok(Self { inner })
    }

    fn poll_events(&mut self) -> Result<Vec<PlatformEvent>> {
        warn!("MacosPlatform::poll_events() called but not used in actor architecture");
        Ok(vec![])
    }

    fn dispatch_actions(
        &mut self,
        _actions: Vec<crate::platform::actions::PlatformAction>,
    ) -> Result<()> {
        warn!("MacosPlatform::dispatch_actions() called but not used in actor architecture");
        Ok(())
    }

    fn get_current_platform_state(&self) -> PlatformState {
        self.inner.get_current_platform_state()
    }

    fn run(self) -> Result<()> {
        info!("MacosPlatform::run() - Delegating to GenericPlatform");
        self.inner.run()
    }

    fn cleanup(&mut self) -> Result<()> {
        info!("MacosPlatform::cleanup() - No cleanup needed (handled by Drop)");
        Ok(())
    }
}
