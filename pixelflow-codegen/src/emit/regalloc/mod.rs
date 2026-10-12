//! Register allocation for selected functions.
//!
//! [`local`] is the allocator: a function from a selected [`Function`](super::Function)
//! to an allocated one. [`policy`] is what it prefers, and [`resource`] is what it
//! allocates from: the pool, the frame, and the tokens they lend.

pub(super) mod local;
mod policy;
pub(super) mod resource;
