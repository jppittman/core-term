//! # The render actor: a scene, a frame, and three priority lanes
//!
//! Rendering a scene is [`Scene::render`](crate::render::scene::Scene::render)
//! — one collapse call per stripe, stripes across threads. This module is what
//! drives it *asynchronously*: an actor that owns the pause/resume and
//! thread-count state, takes a [`RenderRequest`] (scene + frame + the caller's
//! own payload) on its Data lane, and hands the frame back on a
//! [`RenderResponse`] whether or not it drew into it.
//!
//! The split is the one `vsync_actor.rs` uses: [`RenderCore`](actor::RenderCore)
//! is the pure decision core, table-testable with no scheduler in the loop, and
//! [`RendererActor`] is the thin adapter that owns the response channel and the
//! bootstrap handshake.

pub mod actor;
pub mod messages;

pub use actor::RendererActor;
pub use messages::{
    RenderConfig, RenderControl, RenderManagement, RenderRequest, RenderResponse, RendererHandle,
    RendererSetup, RendererSetupHandle,
};
