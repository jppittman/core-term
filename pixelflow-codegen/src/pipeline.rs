//! The driver: from an arena and a lattice's shape to machine code.
//!
//! The stages are [`Pass`] values, run as one chain: the arena is legalized
//! into the lattice's folds ([`Legalize`], `pixelflow_ir::passes::legalize`),
//! lowered to a flat schedule ([`Lower`]), split into its nest ([`Nest`]), its
//! arms' ownership read off backward ([`Own`]) and their blocks laid out
//! ([`Layout`]); then the emitter selects, allocates and assembles it
//! ([`emit::native`]). Nothing here is the emitter's: `emit/` is handed the
//! stages above it as a value and names none of them.

use crate::emit::{self, CompileResult};
use crate::error::CompileError;
use crate::pass::Pass;
use crate::program::{Def, Layout, Nest, Own, arena_to_schedule};
use core::marker::PhantomData;
use pixelflow_ir::LatticeShape;
use pixelflow_ir::arena::{ExprArena, ExprId, UniformDecl, UniformId, UniformIdentity};
use pixelflow_ir::passes::lattice::{Collapse, Domain};

/// The two per-call scalars every collapse reads: where the lattice's sample
/// `(0, 0)` lies, `x0` then `y0`.
///
/// Declared as uniforms by `passes::lattice::collapse`, so the arena names
/// them the way it names any per-call scalar and the emitter loads them the
/// way it loads any uniform — once per call, broadcast. What is particular
/// to them is *where*: not in the link's block, whose layout is the
/// caller's, but in a block of their own, the context entry after the
/// link's (see `KernelFn`). One identity
/// per axis for the whole process, minted once, so every arena declares the
/// same two instances and [`origin_slots`] can find them by identity
/// afterwards.
fn origin() -> [UniformDecl; 2] {
    static ORIGIN: std::sync::OnceLock<[UniformDecl; 2]> = std::sync::OnceLock::new();
    *ORIGIN.get_or_init(|| {
        [0.0, 0.0].map(|default| UniformDecl {
            id: UniformIdentity::mint(),
            default,
        })
    })
}

/// The uniform slots [`origin`]'s two instances hold in a legalized arena —
/// the two `passes::lattice::collapse` declared, which is why they are always
/// present.
fn origin_slots(arena: &ExprArena) -> [UniformId; 2] {
    origin().map(|decl| {
        let slot = arena
            .uniforms()
            .iter()
            .position(|d| d.id == decl.id)
            .unwrap_or_else(|| panic!("a legalized arena declares the origin; this one does not"));
        UniformId(slot as u64)
    })
}

/// `passes::legalize` for the collapse of a lattice of `shape` on a target of
/// `lanes` lanes: the kernel comes back wrapped in the lattice's folds.
struct Legalize<'a> {
    collapse: Collapse,
    arena: PhantomData<&'a ExprArena>,
}

impl Legalize<'_> {
    fn new(shape: LatticeShape, lanes: u32) -> Self {
        let domain = Domain {
            shape,
            origin: origin(),
        };
        Self {
            collapse: Collapse { domain, lanes },
            arena: PhantomData,
        }
    }
}

impl<'a> Pass for Legalize<'a> {
    type Input = (&'a ExprArena, ExprId);
    type Output = (ExprArena, ExprId);

    fn run(self, (arena, root): Self::Input) -> Result<Self::Output, CompileError> {
        pixelflow_ir::passes::legalize(arena, root, &self.collapse).map_err(CompileError::Legalize)
    }
}

/// `arena_to_schedule`: a legalized arena to the flat schedule the later
/// stages read.
struct Lower;

impl Pass for Lower {
    type Input = (ExprArena, ExprId);
    type Output = Vec<Def>;

    fn run(self, (arena, root): Self::Input) -> Result<Self::Output, CompileError> {
        Ok(arena_to_schedule(&arena, root, origin_slots(&arena)))
    }
}

/// Compile an [`ExprArena`] DAG into a
/// **collapse** kernel for a lattice of `shape`: the kernel is wrapped in the
/// lattice's folds by
/// [`pixelflow_ir::passes::legalize`], and every fold is emitted as a loop
/// inside the code — one call fills the whole extent with no per-row or
/// per-batch Rust↔JIT boundary. Matches the
/// `KernelFn` ABI `(ctx, out, pitch)`.
///
/// The context is one base pointer per declared buffer, in the arena's slot
/// order, followed by the uniform block's base pointer (`f32` values in the
/// arena's uniform-slot order, read once per call) and then the origin
/// block's: `x0`, `y0`.
///
/// # Panics
///
/// Panics if the arena names a retired coordinate axis (`Var(2)`/`Var(3)`,
/// the old Z and W). This is the boundary the check belongs on, because it
/// is the *only* one every route to machine code passes through — the
/// shape-keyed cache is one caller, and the benchmark harnesses, the corpus
/// tools and several tests come straight here. `collapse` substitutes only
/// `X` and `Y`, so a retired axis would survive into the schedule as a `Var`
/// no fold binds, and the allocator's refusal there names a binder, not an
/// axis; this one names the axis. Also panics on a zero extent, which has
/// no sample to compute.
pub fn compile(
    arena: &ExprArena,
    root: ExprId,
    shape: LatticeShape,
) -> Result<CompileResult, CompileError> {
    assert!(
        arena.retired_axis(root).is_none(),
        "emit::compile: the arena names Var({:?}), a coordinate axis a \
         lattice no longer has; a per-call scalar is a Uniform",
        arena.retired_axis(root)
    );
    assert!(
        shape.extent().iter().all(|&e| e > 0),
        "emit::compile: degenerate extent {:?}",
        shape.extent()
    );
    let front = |lanes| {
        Legalize::new(shape, lanes)
            .then(Lower)
            .then(Nest)
            .then(Own)
            .then(Layout)
    };
    emit::native(front, (arena, root))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The stages that make a schedule, at a width a test chooses: everything
    /// a compile runs before scoping.
    pub(crate) fn schedule_for(
        a: &ExprArena,
        root: ExprId,
        shape: LatticeShape,
        lanes: u32,
    ) -> Vec<Def> {
        let schedule = Legalize::new(shape, lanes).then(Lower);
        schedule.run((a, root)).expect("legalize")
    }

    /// The backstop for an arena that never became a `Kernel`.
    /// `Kernel::from_parts` refuses an arena naming a retired axis
    /// (`pixelflow-ir`'s `an_arena_naming_a_retired_axis_is_not_a_kernel`),
    /// but `compile` takes a bare arena: the benchmark harnesses and corpus
    /// tools call it directly, and a rewrite rule that instantiated one of
    /// its own metavariables would hand it one past that refusal. So the
    /// arena goes straight in here, and the refusal is this function's own.
    #[test]
    #[should_panic(expected = "emit::compile: the arena names Var")]
    fn an_arena_naming_a_retired_axis_never_reaches_the_emitter() {
        let mut a = pixelflow_ir::arena::ExprArena::new();
        let x = a.push_var(0);
        let z = a.push_var(2);
        let root = a.push_binary(pixelflow_ir::OpKind::Add, x, z);
        let _refused = super::compile(&a, root, LatticeShape::new([4, 4]));
    }
}
