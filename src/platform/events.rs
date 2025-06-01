use anyhow::Error;

#[derive(Debug)]
pub enum SystemEvent {
    PrimaryIoReady,
    UiInputReady,
    Tick,
    Error(Error),
    ShutdownAdvised,
}
