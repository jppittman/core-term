//! # Scopes
//!
//! The one definition of which binding a use of a name refers to.
//!
//! Rust's lexical scoping, exactly. A `let` is visible from the statement after
//! it to the end of its block; it shadows every binding of its name already in
//! scope; and a use sees the innermost binding in scope. [`Scopes`] is that
//! rule, and both `sema` (over its symbols) and lowering (over arena ids)
//! resolve names through it, so the two cannot disagree about which binding a
//! name means.
//!
//! One map per scope, not one map for the kernel: a block that ends must give
//! back exactly what it shadowed (deleting the name would lose an outer
//! parameter), and an inner `let` must not replace an outer one for the rest
//! of the kernel.

use crate::ast::Name;
use std::collections::HashMap;

/// Rust's lexical scoping over bindings of `V`.
///
/// Sema binds its symbols and lowering binds arena ids; both resolve a name
/// through this, which is what makes them agree on what it refers to.
#[derive(Debug, Clone)]
pub(crate) struct Scopes<V> {
    /// The function's own scope — its coordinates, parameters and the
    /// block's items. It lasts as long as the table and is not a block, so no
    /// pop reaches it.
    function: HashMap<Name, V>,
    /// One scope per enclosing block, innermost last.
    blocks: Vec<HashMap<Name, V>>,
}

impl<V> Default for Scopes<V> {
    fn default() -> Self {
        Scopes {
            function: HashMap::new(),
            blocks: Vec::new(),
        }
    }
}

impl<V> Scopes<V> {
    /// Open a block's scope.
    pub(crate) fn push_scope(&mut self) {
        self.blocks.push(HashMap::new());
    }

    /// Close the innermost block's scope. Its bindings go out of scope, and
    /// whatever they shadowed is visible again.
    pub(crate) fn pop_scope(&mut self) {
        self.blocks.pop();
    }

    /// Bind `name` in the innermost scope, shadowing any binding of it
    /// already in scope — including an earlier one in the same block.
    pub(crate) fn bind(&mut self, name: Name, value: V) {
        self.blocks
            .last_mut()
            .unwrap_or(&mut self.function)
            .insert(name, value);
    }

    /// The innermost binding of `name` in scope.
    pub(crate) fn lookup(&self, name: &Name) -> Option<&V> {
        self.blocks
            .iter()
            .rev()
            .find_map(|block| block.get(name))
            .or_else(|| self.function.get(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proc_macro2::Span;
    use syn::Ident;

    fn name(text: &str) -> Name {
        Name::of(&Ident::new(text, Span::call_site()))
    }

    #[test]
    fn a_lookup_sees_the_innermost_binding_in_scope() {
        let mut scopes = Scopes::default();
        scopes.bind(name("a"), 0);
        scopes.push_scope();
        scopes.bind(name("a"), 1);
        scopes.push_scope();
        assert_eq!(scopes.lookup(&name("a")), Some(&1));
        scopes.bind(name("a"), 2);
        scopes.bind(name("a"), 3);
        assert_eq!(scopes.lookup(&name("a")), Some(&3));
        scopes.pop_scope();
        assert_eq!(scopes.lookup(&name("a")), Some(&1));
        scopes.pop_scope();
        assert_eq!(scopes.lookup(&name("a")), Some(&0));
    }
}
