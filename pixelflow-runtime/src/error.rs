use thiserror::Error;

#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("Driver channel disconnected")]
    DriverChannelDisconnected,
    #[error("Only original driver can run (this is a clone)")]
    DriverCloneError,
    #[error("Expected CreateWindow, got {0:?}")]
    UnexpectedCommand(String),
    #[error("Atomics error: {0}")]
    AtomicsError(String),
    #[error("Atomics store failed: {0}")]
    AtomicsStoreError(String),
    #[error("Atomics notify failed: {0}")]
    AtomicsNotifyError(String),
    #[error("RingBuffer full")]
    RingBufferFull,
    #[error("Web resources not initialized. Call init_resources() first.")]
    WebResourcesNotInitialized,
    #[error("Failed to get 2d context")]
    WebContextError,
    #[error("Context is null")]
    WebContextNull,
    #[error("Failed to cast context")]
    WebContextCastError,
    #[error("Failed to create ImageData: {0}")]
    WebImageDataError(String),
    #[error("Failed to put image data: {0}")]
    WebPutImageError(String),
    #[error("XInitThreads failed")]
    XInitThreadsFailed,
    #[error("Failed to open X display")]
    XOpenDisplayFailed,
    #[error("XCreateImage failed")]
    XCreateImageFailed,
    #[error("Failed to create Metal device")]
    MetalDeviceError,
    #[error("Failed to send event to application: {0}")]
    EventSendError(String),
    #[error("Failed to init: {0}")]
    InitError(String),
}
