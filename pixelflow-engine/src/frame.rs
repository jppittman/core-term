//! Zero-copy frame packet infrastructure for v11.0 architecture.
//!
//! This module provides the `FramePacket<T>` type for transferring surfaces
//! between the logic thread and render thread without copying pixel data.

use pixelflow_core::pipe::Surface;
use pixelflow_core::Batch;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;

/// A frame packet containing a composed surface and recycle channel.
///
/// The packet transfers ownership of the surface to the engine for rendering,
/// then returns via the recycle channel for reuse (zero-copy ping-pong).
///
/// The `recycle_tx` is wrapped in Arc - this is the "ghetto borrow" pattern
/// that works across thread boundaries. The Arc ensures we're explicit about
/// shared ownership of the return channel.
///
/// # Type Parameters
/// * `T` - The surface type (e.g., `TerminalSurface`)
pub struct FramePacket<T>
where
    T: Surface<u32> + Send,
{
    /// The composed surface to render.
    pub surface: T,

    /// Channel for returning the packet after rendering (Arc-wrapped).
    pub recycle_tx: Arc<SyncSender<FramePacket<T>>>,
}

impl<T> std::fmt::Debug for FramePacket<T>
where
    T: Surface<u32> + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FramePacket")
            .field("surface", &"<surface>")
            .field("recycle_tx", &"<channel>")
            .finish()
    }
}

impl<T> FramePacket<T>
where
    T: Surface<u32> + Send,
{
    /// Creates a new frame packet with the given surface and recycle channel.
    pub fn new(surface: T, recycle_tx: Arc<SyncSender<FramePacket<T>>>) -> Self {
        Self {
            surface,
            recycle_tx,
        }
    }

    /// Recycles this packet back to the logic thread.
    ///
    /// Consumes self and sends it through the recycle channel.
    /// The Arc clone is just a refcount bump - no data copied.
    pub fn recycle(self) {
        let tx = Arc::clone(&self.recycle_tx);
        let _ = tx.send(self);
    }
}

/// Handle for submitting frames to the engine from the logic thread.
///
/// This is the "producer" side of the channel.
pub struct EngineHandle<T>
where
    T: Surface<u32> + Send,
{
    submit_tx: SyncSender<FramePacket<T>>,
}

impl<T> EngineHandle<T>
where
    T: Surface<u32> + Send,
{
    /// Creates a new engine handle with the given submit channel.
    pub fn new(submit_tx: SyncSender<FramePacket<T>>) -> Self {
        Self { submit_tx }
    }

    /// Submits a frame packet for rendering.
    ///
    /// This call may block if the channel buffer is full (back-pressure).
    pub fn submit_frame(&self, packet: FramePacket<T>) -> Result<(), FramePacket<T>> {
        self.submit_tx.send(packet).map_err(|e| e.0)
    }

    /// Tries to submit a frame packet without blocking.
    ///
    /// Returns the packet back if the channel is full.
    pub fn try_submit_frame(&self, packet: FramePacket<T>) -> Result<(), FramePacket<T>> {
        self.submit_tx.try_send(packet).map_err(|e| match e {
            std::sync::mpsc::TrySendError::Full(p) => p,
            std::sync::mpsc::TrySendError::Disconnected(p) => p,
        })
    }
}

impl<T> Clone for EngineHandle<T>
where
    T: Surface<u32> + Send,
{
    fn clone(&self) -> Self {
        Self {
            submit_tx: self.submit_tx.clone(),
        }
    }
}

/// Creates a channel pair for frame submission.
///
/// Returns (handle, receiver) where:
/// - `handle` is used by the logic thread to submit frames
/// - `receiver` is used by the engine to receive frames
///
/// The channel has a buffer of 1 slot for ping-pong operation.
pub fn create_frame_channel<T>() -> (EngineHandle<T>, Receiver<FramePacket<T>>)
where
    T: Surface<u32> + Send,
{
    let (tx, rx) = sync_channel(1);
    (EngineHandle::new(tx), rx)
}

/// Creates a recycle channel for returning packets to the logic thread.
///
/// Returns (sender, receiver) where:
/// - `sender` is Arc-wrapped and cloned into each FramePacket
/// - `receiver` is held by the logic thread to get packets back
pub fn create_recycle_channel<T>() -> (Arc<SyncSender<FramePacket<T>>>, Receiver<FramePacket<T>>)
where
    T: Surface<u32> + Send,
{
    let (tx, rx) = sync_channel(2); // 2 slots for double-buffering
    (Arc::new(tx), rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal test surface
    #[derive(Clone, Copy)]
    struct TestSurface {
        color: u32,
    }

    impl Surface<u32> for TestSurface {
        fn eval(&self, _x: Batch<u32>, _y: Batch<u32>) -> Batch<u32> {
            Batch::splat(self.color)
        }
    }

    #[test]
    fn test_create_channels() {
        let (_handle, _rx) = create_frame_channel::<TestSurface>();
        let (_recycle_tx, _recycle_rx) = create_recycle_channel::<TestSurface>();
    }

    #[test]
    fn test_submit_and_receive() {
        let (handle, rx) = create_frame_channel::<TestSurface>();
        let (recycle_tx, _recycle_rx) = create_recycle_channel::<TestSurface>();

        let surface = TestSurface { color: 0xFF00FF00 };
        let packet = FramePacket::new(surface, recycle_tx);

        handle.submit_frame(packet).unwrap();

        let received = rx.recv().unwrap();
        assert_eq!(received.surface.color, 0xFF00FF00);
    }

    #[test]
    fn test_recycle_loop() {
        let (handle, rx) = create_frame_channel::<TestSurface>();
        let (recycle_tx, recycle_rx) = create_recycle_channel::<TestSurface>();

        // Create and submit a packet
        let surface = TestSurface { color: 0xFFFF0000 };
        let packet = FramePacket::new(surface, Arc::clone(&recycle_tx));
        handle.submit_frame(packet).unwrap();

        // Receive and "render" it
        let received = rx.recv().unwrap();
        assert_eq!(received.surface.color, 0xFFFF0000);

        // Recycle using the method (clean API)
        received.recycle();

        // Logic thread receives the recycled packet
        let recycled = recycle_rx.recv().unwrap();
        assert_eq!(recycled.surface.color, 0xFFFF0000);
    }
}
