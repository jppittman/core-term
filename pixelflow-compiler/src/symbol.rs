//! # Symbol Table
//!
//! The names a kernel body can see, and the one definition of which binding a
//! use of a name refers to.
//!
//! ## Symbol Classes
//!
//! | Class      | Binding Time        | Arena Representation | Example |
//! |------------|---------------------|----------------------|---------|
//! | Intrinsic  | Collapse time       | `Var(0)`, `Var(1)`   | X, Y    |
//! | Parameter  | Per call (uniform)  | `Uniform(slot)`, one per record field | cx, p |
//! | Structural | Instantiation       | a count; `N as f32` is `Const(N)` | N |
//! | Const      | Expansion time      | `Const(v)`, or a count | PI, N |
//! | Local      | Expression scope    | A shared `ExprId`, or a record's fields | dx, q |
//! | Index      | A fold's body       | The fold's binder `Var` | i    |
//!
//! A helper's parameters are a further thing at lowering — the argument's
//! own node, or a record argument's fields, bound by name where the helper
//! is inlined — but to `sema` they are parameters like any other: a name
//! with a type.
//!
//! A kernel-typed parameter, `impl Fn(f32, f32) -> f32` (Phase D-a), is a
//! `Parameter` of type [`Ty::Kernel`]. To `sema` it is a name that is
//! applied, `k(x, y)`, or passed on; at lowering it is an argument admitted
//! when the entry's host function is called, whose term each application
//! splices in at its coordinates.
//!
//! ## Intrinsic Coordinates
//!
//! The intrinsic coordinates X and Y are special:
//! - They become `Var(0)` and `Var(1)` arena nodes
//! - They are in scope in an entry, bound like any other name: this table
//!   would let a parameter or a `let` shadow them, so `sema` is what refuses
//!   a parameter or a `let` named X or Y (and a use of one in a helper)
//!   before anything is registered
//!
//! ## Parameter Symbols
//!
//! An entry's parameters are its uniforms
//! (docs/language/2026-09-25-the-language-is-kernel.md §1.4): `|cx: f32, cy:
//! f32|` produces `move |cx: f32, cy: f32| -> Kernel`, whose kernel declares
//! one uniform per parameter with the call's value as its default, and a
//! reference in the body is that uniform's `Uniform` leaf. A record
//! parameter is one uniform per field. An entry's structural parameters
//! (`const N: usize`) are counts, fixed per instantiation of its host
//! function.
//!
//! ## Scoping
//!
//! Rust's lexical scoping, exactly. A `let` is visible from the statement after
//! it to the end of its block; it shadows every binding of its name already in
//! scope; and a use sees the innermost binding in scope. [`Scopes`] is that
//! rule, and both `sema` (over [`Symbol`]s) and lowering (over arena ids)
//! resolve names through it, so the two cannot disagree about which binding a
//! name means.
//!
//! One map per scope, not one map for the kernel: a block that ends must give
//! back exactly what it shadowed (deleting the name would lose an outer
//! parameter), and an inner `let` must not replace an outer one for the rest
//! of the kernel.

use crate::ast::Name;
use crate::sema::Ty;
use pixelflow_ir::arena::Axis;
use std::collections::HashMap;

/// The binding class of a symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SymbolKind {
    /// Intrinsic coordinate variable (X, Y).
    /// Bound at collapse time, as a `Var` node.
    Intrinsic,

    /// A declared parameter (e.g., `r: f32`): an entry's is bound when its
    /// host function runs, a helper's where the helper is inlined.
    Parameter,

    /// A `const` item, evaluated at expansion.
    Const,

    /// An entry's structural parameter (`const N: usize`): a count, fixed
    /// when its host function is instantiated.
    Structural,

    /// Local variable introduced by `let`.
    /// Scoped to the containing block.
    Local,

    /// A fold's index, the closure's parameter: a `usize`, scoped to the
    /// fold's body.
    Index,
}

/// A symbol in the symbol table.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Symbol {
    /// What kind of symbol this is.
    pub kind: SymbolKind,
    /// The symbol's type.
    pub ty: Ty,
}

/// Rust's lexical scoping over bindings of `V`.
///
/// Sema binds [`Symbol`]s and lowering binds arena ids; both resolve a name
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

/// The symbol table for one function's analysis.
#[derive(Debug, Clone)]
pub(crate) struct SymbolTable {
    symbols: Scopes<Symbol>,
}

impl SymbolTable {
    /// The coordinate intrinsics, by spelling. A lattice has two axes; `Z`
    /// and `W` are refused by name in sema, with the message that points at
    /// uniforms.
    const COORDINATES: [(&'static str, Axis); 2] = [("X", Axis::X), ("Y", Axis::Y)];

    /// The axis a name is the coordinate of, if it is one.
    pub(crate) fn coordinate(name: &Name) -> Option<Axis> {
        Self::COORDINATES
            .into_iter()
            .find(|&(spelling, _)| name == spelling)
            .map(|(_, axis)| axis)
    }

    /// Create a new symbol table with intrinsic coordinates pre-populated.
    pub(crate) fn new() -> Self {
        let mut table = SymbolTable {
            symbols: Scopes::default(),
        };
        for (spelling, _) in Self::COORDINATES {
            table.symbols.bind(
                Name::spelled(spelling),
                Symbol {
                    kind: SymbolKind::Intrinsic,
                    ty: Ty::F32,
                },
            );
        }
        table
    }

    /// Register a parameter symbol (e.g., `r: f32`).
    pub(crate) fn register_parameter(&mut self, name: Name, ty: Ty) {
        self.bind(name, SymbolKind::Parameter, ty);
    }

    /// Register a `const` item: an `f32`, or a `usize` count.
    pub(crate) fn register_const(&mut self, name: Name, ty: Ty) {
        self.bind(name, SymbolKind::Const, ty);
    }

    /// Register an entry's structural parameter, a `usize` count.
    pub(crate) fn register_structural(&mut self, name: Name) {
        self.bind(name, SymbolKind::Structural, Ty::Usize);
    }

    /// Register a local variable in the innermost scope.
    pub(crate) fn register_local(&mut self, name: Name, ty: Ty) {
        self.bind(name, SymbolKind::Local, ty);
    }

    /// Register a fold's index, a `usize`, in the innermost scope: the
    /// scope the fold's body is typed in.
    pub(crate) fn register_index(&mut self, name: Name) {
        self.bind(name, SymbolKind::Index, Ty::Usize);
    }

    fn bind(&mut self, name: Name, kind: SymbolKind, ty: Ty) {
        self.symbols.bind(name, Symbol { kind, ty });
    }

    /// Look up the innermost binding of a name in scope.
    pub(crate) fn lookup(&self, name: &Name) -> Option<&Symbol> {
        self.symbols.lookup(name)
    }

    /// Check if a name is an intrinsic coordinate.
    pub(crate) fn is_intrinsic(&self, name: &Name) -> bool {
        self.lookup(name)
            .is_some_and(|s| s.kind == SymbolKind::Intrinsic)
    }

    /// Open a block's scope.
    pub(crate) fn push_scope(&mut self) {
        self.symbols.push_scope();
    }

    /// Close the innermost block's scope, and with it every local bound in it.
    pub(crate) fn pop_scope(&mut self) {
        self.symbols.pop_scope();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(text: &str) -> Name {
        Name::spelled(text)
    }

    #[test]
    fn intrinsics_are_predefined() {
        let table = SymbolTable::new();
        assert!(table.is_intrinsic(&name("X")));
        assert!(table.is_intrinsic(&name("Y")));
        assert!(!table.is_intrinsic(&name("Z")));
        assert!(!table.is_intrinsic(&name("W")));
        assert!(!table.is_intrinsic(&name("cx")));
    }

    #[test]
    fn register_parameter_marks_name_as_parameter_and_not_intrinsic() {
        let mut table = SymbolTable::new();
        table.register_parameter(name("radius"), Ty::F32);

        assert_eq!(
            table.lookup(&name("radius")).map(|s| s.kind),
            Some(SymbolKind::Parameter)
        );
        assert!(!table.is_intrinsic(&name("radius")));
        assert_eq!(table.lookup(&name("radius")).map(|s| s.ty), Some(Ty::F32));
    }

    #[test]
    fn pop_scope_removes_locals_registered_since_the_matching_push_scope() {
        let mut table = SymbolTable::new();
        table.push_scope();
        table.register_local(name("dx"), Ty::F32);
        assert!(table.lookup(&name("dx")).is_some());

        table.pop_scope();

        assert!(table.lookup(&name("dx")).is_none());
        // The outer scope (and its intrinsics) must be untouched.
        assert!(table.is_intrinsic(&name("X")));
    }

    /// Popping a scope restores what its bindings shadowed: a local shadowing
    /// a parameter does not delete the parameter when the block ends.
    #[test]
    fn pop_scope_restores_the_binding_a_local_shadowed() {
        let mut table = SymbolTable::new();
        table.register_parameter(name("r"), Ty::F32);
        table.push_scope();
        table.register_local(name("r"), Ty::Bool);
        assert_eq!(
            table.lookup(&name("r")).map(|s| s.kind),
            Some(SymbolKind::Local)
        );
        assert_eq!(table.lookup(&name("r")).map(|s| s.ty), Some(Ty::Bool));

        table.pop_scope();

        assert_eq!(
            table.lookup(&name("r")).map(|s| s.kind),
            Some(SymbolKind::Parameter)
        );
        assert_eq!(table.lookup(&name("r")).map(|s| s.ty), Some(Ty::F32));
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
