//! # Combinators Module
//!
//! Higher-level constructs: Select (conditionals) and Fix (iteration).

mod fix;
mod select;

pub use fix::Fix;
pub use select::Select;
