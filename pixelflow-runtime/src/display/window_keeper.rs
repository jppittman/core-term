//! The driver's ownership of the window buffer.
//!
//! The buffer is **pulled, not pushed**: this type allocates it, keeps it at rest, lends it out
//! on request, and takes it back after presentation. See §8 of
//! `docs/designs/pixelflow-runtime-engine-mesh-migration.md` for why that direction rather than
//! the reverse — briefly, the OS decides the size, and the intended zero-copy endgame has the
//! renderer writing into memory only the driver can obtain.
//!
//! # Why this is platform-agnostic
//!
//! Nothing here is X11- or Cocoa-specific: a buffer is a [`Frame`] of platform pixels, and
//! allocating one needs a width and a height. Keeping it out of the `PlatformOps` impls means
//! the policy is written once rather than twice, and in particular means it is not written a
//! second time in the macOS half, which cannot be compiled on a Linux dev box. The ops report
//! the geometry the OS gave them; this decides what buffer exists.

use super::messages::{Generation, Surface, Window};
use crate::pixel::PlatformPixel;
use pixelflow_graphics::render::Frame;

/// What the driver should do with a buffer handed back to be presented.
///
/// Returned rather than acted on, because the blit itself is platform-specific and lives in the
/// `PlatformOps` impl. Making the two outcomes a type means the superseded case has to be
/// written down at the call site instead of being the branch someone forgets.
#[derive(Debug)]
#[must_use = "the buffer is inside; dropping this destroys the only framebuffer there is"]
pub enum Presented {
    /// Still the buffer the driver wants. Blit it, then give it back with [`WindowKeeper::rest`].
    Blit(Window),
    /// A resize replaced this buffer while it was out. Its pixels are internally consistent but
    /// the native window is a different shape now, so blitting it would show a stale-sized
    /// frame. The buffer is dropped here — by its owner, deliberately, which is the only way a
    /// buffer is ever allowed to die.
    Superseded,
}

/// Owns the window's frame buffer on behalf of the driver.
#[derive(Debug, Default)]
pub struct WindowKeeper {
    /// The buffer, when nobody is drawing into it. `None` means it is out on loan.
    resting: Option<Window>,
    /// The newest buffer this has allocated. Anything older is a buffer a resize has replaced,
    /// wherever it currently is.
    generation: Generation,
    /// Somebody asked for the buffer and has not been given one yet.
    ///
    /// An earlier revision refused to hold this, on the grounds that a driver keeping a request
    /// would also have to decide *when* to reconsider it, which is the renderer's scheduling
    /// policy. That was wrong twice over. There is exactly one moment at which the answer can
    /// change — the buffer becoming free — and the driver is the only actor that knows it.
    /// Without this, the request and the buffer's return travel different lanes, so a request
    /// sent immediately after a `Present` is drained *before* it and refused; every frame then
    /// waits for the next tick.
    wanted: bool,
}

impl WindowKeeper {
    /// No window yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The platform reported a window at this geometry — create the buffer for it.
    ///
    /// Used for the initial creation and for every resize alike, because they are the same
    /// event as far as the buffer is concerned: the old one, if any, is now the wrong size.
    /// Allocating immediately rather than deferring until the outstanding buffer comes back is
    /// what lets the very next request be granted at the *new* size; the old buffer is dropped
    /// when it is handed back, and recognised as stale by its generation.
    pub fn surface_changed(&mut self, surface: Surface) {
        self.generation = self.generation.next();
        self.resting = Some(Window {
            id: surface.id,
            // The lattice, not the layout: allocating at the point extent would render at half
            // density on a Retina display and be invisible on any 1:1 monitor.
            frame: Frame::<PlatformPixel>::new(surface.frame_width, surface.frame_height),
            width_px: surface.width_px,
            height_px: surface.height_px,
            scale: surface.scale,
            generation: self.generation,
        });
    }

    /// Somebody wants the buffer. Answered by [`pending_grant`](Self::pending_grant), now or as
    /// soon as one is free.
    ///
    /// Repeat requests collapse into the one outstanding want, which is why the renderer may
    /// re-ask freely — there is one buffer, so there can only ever be one grant.
    pub fn request(&mut self) {
        self.wanted = true;
    }

    /// The buffer to hand out right now, if one is free and somebody is waiting for it.
    ///
    /// Call after anything that can change either half of that: a request arriving, a buffer
    /// coming back, a resize allocating a new one.
    #[must_use = "this is the grant; dropping it strands the renderer until its next request"]
    pub fn pending_grant(&mut self) -> Option<Window> {
        if !self.wanted {
            return None;
        }
        let window = self.resting.take()?;
        self.wanted = false;
        Some(window)
    }

    /// Decide what to do with a buffer handed back for presentation.
    pub fn presented(&mut self, window: Window) -> Presented {
        if window.generation == self.generation {
            Presented::Blit(window)
        } else {
            log::debug!(
                "Discarding superseded buffer {} ({}x{}) - buffer {} is current",
                window.generation,
                window.width_px,
                window.height_px,
                self.generation
            );
            Presented::Superseded
        }
    }

    /// Put a buffer back at rest after it has been blitted.
    pub fn rest(&mut self, window: Window) {
        // A resize can land between `presented` and here, in which case this buffer is already
        // superseded and `resting` holds the new one. Overwriting it would put the wrong-sized
        // buffer back into circulation with no way to notice.
        if window.generation != self.generation {
            return;
        }
        debug_assert!(
            self.resting.is_none(),
            "two live buffers of generation {}: one resting, one returned",
            window.generation
        );
        self.resting = Some(window);
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    /// Request the buffer and take whatever answer is available immediately — the two-step the
    /// driver performs on every `RequestWindow`, as one call so the tests read as asks.
    fn ask(keeper: &mut WindowKeeper) -> Option<Window> {
        keeper.request();
        keeper.pending_grant()
    }

    /// A 1:1 surface, as X11 and a non-Retina Mac report.
    fn flat(width: u32, height: u32) -> Surface {
        Surface {
            id: super::super::messages::WindowId(1),
            width_px: width,
            height_px: height,
            frame_width: width,
            frame_height: height,
            scale: 1.0,
        }
    }

    #[test]
    fn a_buffer_is_lent_once_and_comes_back() {
        let mut keeper = WindowKeeper::new();
        keeper.surface_changed(flat(4, 4));

        let window = ask(&mut keeper).expect("the buffer is resting");
        assert!(
            ask(&mut keeper).is_none(),
            "there is exactly one buffer, so a second borrower gets nothing"
        );

        let Presented::Blit(window) = keeper.presented(window) else {
            panic!("nothing resized, so this is still the current buffer");
        };
        keeper.rest(window);
        assert!(ask(&mut keeper).is_some(), "and it is available again");
    }

    #[test]
    fn a_resize_while_the_buffer_is_out_supersedes_it() {
        // The one identity check the pull protocol keeps (§8). Binding the kernel to the frame
        // makes the pixels correct *for that buffer*; it cannot make an old-sized buffer correct
        // for a window that is now a different shape.
        let mut keeper = WindowKeeper::new();
        keeper.surface_changed(flat(4, 4));
        let old = ask(&mut keeper).expect("the buffer is resting");

        keeper.surface_changed(flat(8, 8));

        assert!(
            matches!(keeper.presented(old), Presented::Superseded),
            "an old-size buffer must not be blitted onto a resized window"
        );
        let fresh = ask(&mut keeper).expect("the resize already allocated the new buffer");
        assert_eq!(
            (fresh.width_px, fresh.height_px),
            (8, 8),
            "and the next borrower gets the new size without waiting for the old buffer"
        );
    }

    #[test]
    fn a_resize_between_presentation_and_rest_does_not_restore_the_old_buffer() {
        // `presented` and `rest` are two steps because the blit in between is platform-specific,
        // and a resize can land in that gap. Putting the blitted buffer back unconditionally
        // would overwrite the correctly-sized one the resize just allocated.
        let mut keeper = WindowKeeper::new();
        keeper.surface_changed(flat(4, 4));
        let old = ask(&mut keeper).expect("the buffer is resting");

        let Presented::Blit(old) = keeper.presented(old) else {
            panic!("current at the time it was handed back");
        };
        keeper.surface_changed(flat(8, 8));
        keeper.rest(old);

        let fresh = ask(&mut keeper).expect("a buffer is resting");
        assert_eq!(
            (fresh.width_px, fresh.height_px),
            (8, 8),
            "the resize's buffer must survive a late return of the old one"
        );
    }

    #[test]
    fn a_retina_surface_allocates_the_lattice_not_the_layout() {
        // The bug this type's `Surface` parameter exists to make unrepresentable: on a 2x
        // display the buffer must be twice the point extent per axis. Allocating at the point
        // extent renders at half density — and is invisible on any 1:1 monitor, which is every
        // monitor a Linux dev box is likely to have.
        let mut keeper = WindowKeeper::new();
        keeper.surface_changed(Surface {
            id: super::super::messages::WindowId(1),
            width_px: 400,
            height_px: 300,
            frame_width: 800,
            frame_height: 600,
            scale: 2.0,
        });

        let window = ask(&mut keeper).expect("the buffer is resting");
        assert_eq!(
            (window.frame.width, window.frame.height),
            (800, 600),
            "the buffer is the sample lattice"
        );
        assert_eq!(
            (window.width_px, window.height_px),
            (400, 300),
            "and it still reports the point extent the scene is authored in"
        );
    }

    #[test]
    fn nothing_is_lent_before_a_window_exists() {
        let mut keeper = WindowKeeper::new();
        assert!(ask(&mut keeper).is_none());
    }

    /// The reason the keeper remembers a request at all: the buffer's return and the next
    /// request travel different lanes, and the request is drained first. Refusing it outright
    /// costs a frame on *every* cycle, not just under contention.
    #[test]
    fn a_request_refused_while_the_buffer_is_out_is_answered_when_it_returns() {
        let mut keeper = WindowKeeper::new();
        keeper.surface_changed(flat(4, 4));
        let window = ask(&mut keeper).expect("the buffer is resting");

        assert!(ask(&mut keeper).is_none(), "it is out; nothing to give yet");

        let Presented::Blit(window) = keeper.presented(window) else {
            panic!("nothing resized");
        };
        keeper.rest(window);
        assert!(
            keeper.pending_grant().is_some(),
            "the outstanding request is answered the moment the buffer is free"
        );
        assert!(
            keeper.pending_grant().is_none(),
            "and answering it consumes it — one request, one grant"
        );
    }

    /// A request that arrives while the buffer is out, then a resize: the waiting renderer gets
    /// the *new* buffer, without having to notice the resize or ask again.
    #[test]
    fn a_pending_request_is_answered_by_a_resize_allocation() {
        let mut keeper = WindowKeeper::new();
        keeper.surface_changed(flat(4, 4));
        let old = ask(&mut keeper).expect("the buffer is resting");
        assert!(ask(&mut keeper).is_none(), "out with the renderer");

        keeper.surface_changed(flat(8, 8));
        let fresh = keeper
            .pending_grant()
            .expect("the resize allocated a buffer and somebody is waiting for one");
        assert_eq!((fresh.width_px, fresh.height_px), (8, 8));

        assert!(
            matches!(keeper.presented(old), Presented::Superseded),
            "and the old one is still recognised as stale when it comes back"
        );
    }
}
