use std::fmt;

#[derive(Debug)]
pub enum RuntimeError {
    DriverChannelDisconnected,
    DriverCloneError,
    UnexpectedCommand(String),
    AtomicsError(String),
    AtomicsStoreError(String),
    AtomicsNotifyError(String),
    RingBufferFull,
    WebResourcesNotInitialized,
    WebContextError,
    WebContextNull,
    WebContextCastError,
    WebImageDataError(String),
    WebPutImageError(String),
    XInitThreadsFailed,
    XOpenDisplayFailed,
    XCreateImageFailed,
    MetalDeviceError,
    EventSendError(String),
    InitError(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DriverChannelDisconnected => write!(f, "Driver channel disconnected"),
            Self::DriverCloneError => write!(f, "Only original driver can run (this is a clone)"),
            Self::UnexpectedCommand(s) => write!(f, "Expected CreateWindow, got {:?}", s),
            Self::AtomicsError(s) => write!(f, "Atomics error: {}", s),
            Self::AtomicsStoreError(s) => write!(f, "Atomics store failed: {}", s),
            Self::AtomicsNotifyError(s) => write!(f, "Atomics notify failed: {}", s),
            Self::RingBufferFull => write!(f, "RingBuffer full"),
            Self::WebResourcesNotInitialized => write!(
                f,
                "Web resources not initialized. Call init_resources() first."
            ),
            Self::WebContextError => write!(f, "Failed to get 2d context"),
            Self::WebContextNull => write!(f, "Context is null"),
            Self::WebContextCastError => write!(f, "Failed to cast context"),
            Self::WebImageDataError(s) => write!(f, "Failed to create ImageData: {}", s),
            Self::WebPutImageError(s) => write!(f, "Failed to put image data: {}", s),
            Self::XInitThreadsFailed => write!(f, "XInitThreads failed"),
            Self::XOpenDisplayFailed => write!(f, "Failed to open X display"),
            Self::XCreateImageFailed => write!(f, "XCreateImage failed"),
            Self::MetalDeviceError => write!(f, "Failed to create Metal device"),
            Self::EventSendError(s) => write!(f, "Failed to send event to application: {}", s),
            Self::InitError(s) => write!(f, "Failed to init: {}", s),
        }
    }
}

impl std::error::Error for RuntimeError {}
