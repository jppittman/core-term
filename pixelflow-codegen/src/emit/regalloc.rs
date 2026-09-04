//! Register allocation for scheduled DAG expressions.
//!
//! Allocation is one algorithm ([`LinearScan`]) parameterised by one
//! description of the target ([`RegisterFile`]). Everything that differs
//! between x86-64 and aarch64 — which registers hold the coordinate inputs,
//! where the allocatable window starts and how wide it is, which fixed
//! registers spilled operands reload into, how many bytes a spilled vector
//! occupies — is a field of that struct and appears nowhere else. Backends
//! declare one `const` and the allocator is architecture-independent.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use pixelflow_ir::kind::OpKind;

use super::{Reg, ScheduledOp};

/// A value in the program (SSA-style).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

/// One step of a schedule: a value, and the operation that defines it.
///
/// Every step defines exactly one value — the DAG is in SSA form — so a
/// schedule is a sequence of these, and a value's program point is its index
/// in that sequence.
#[derive(Clone, Debug)]
pub struct Def {
    /// The value this step defines.
    pub value: ValueId,
    /// The operation that computes it.
    pub op: ScheduledOp,
}

/// The complete platform-dependent surface of register allocation.
///
/// Allocation policy is target-independent; only these numbers are not. A
/// backend declares one of these as a `const` next to its encodings, and
/// [`RegisterFile::checked`] turns a layout that contradicts itself — a reload
/// register inside the allocatable window, say — into a build error rather
/// than a miscompile that shows up as wrong pixels.
///
/// The register file is described once and consulted everywhere: the emitter
/// reads `reload` for its spill choreography and `vector_bytes` for its frame
/// arithmetic, so there is no second copy of any of it to drift.
/// A set of registers from one file, as a bitmask over register numbers.
///
/// The allocatable pool used to be a base plus a count — a contiguous run. On
/// every real target the free registers are *not* contiguous: on SSE2 `xmm14`
/// and `xmm15` sit above the reload pair, and on AVX-512 fifteen free
/// registers sit above it with the gather's scratch in between. A range can only ever name whichever free registers happen to be
/// adjacent, so it silently rounded the pool down to a fraction of the machine.
///
/// A set says the true thing: these registers, whichever they are.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RegSet(u32);

impl RegSet {
    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// The set containing exactly `regs`.
    #[must_use]
    pub const fn of(regs: &[Reg]) -> Self {
        let mut bits = 0u32;
        let mut i = 0;
        while i < regs.len() {
            let r = regs[i].0;
            assert!(
                r < 32,
                "register number out of range for a 32-register file"
            );
            bits |= 1 << r;
            i += 1;
        }
        Self(bits)
    }

    /// The contiguous run `base .. base + count`.
    #[must_use]
    pub const fn range(base: u8, count: u8) -> Self {
        let mut bits = 0u32;
        let mut i = 0;
        while i < count {
            let r = base + i;
            assert!(
                r < 32,
                "register number out of range for a 32-register file"
            );
            bits |= 1 << r;
            i += 1;
        }
        Self(bits)
    }

    /// This set plus every member of `other`.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// This set minus every member of `other`.
    ///
    /// How a scope inside a loop sees the pool: a register carrying a value
    /// across that loop is not available to anything the loop contains.
    #[must_use]
    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    #[must_use]
    pub const fn contains(self, r: Reg) -> bool {
        r.0 < 32 && self.0 & (1 << r.0) != 0
    }

    /// How many registers the set holds.
    #[must_use]
    pub const fn len(self) -> u8 {
        self.0.count_ones() as u8
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The lowest `n` members, or all of them if the set is smaller.
    ///
    /// This is how [`EmitCtx::max_regs`](super::EmitCtx) forces spilling for
    /// pressure testing: it only ever shrinks.
    #[must_use]
    pub const fn take(self, n: u8) -> Self {
        let mut kept = 0u32;
        let mut taken = 0u8;
        let mut r = 0u8;
        while r < 32 {
            if taken < n && self.0 & (1 << r) != 0 {
                kept |= 1 << r;
                taken += 1;
            }
            r += 1;
        }
        Self(kept)
    }

    /// Members low to high.
    pub fn iter(self) -> impl Iterator<Item = Reg> + use<> {
        (0u8..32).filter(move |r| self.0 & (1 << r) != 0).map(Reg)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RegisterFile {
    /// Registers holding the coordinate inputs, in order: X, Y, Z, W.
    ///
    /// A `Var(i)` in the schedule is pre-colored to `inputs[i]`; a `Var` past
    /// the end of this array is a value the target cannot supply.
    pub inputs: [Reg; 4],

    /// Every register the allocator may hand out.
    ///
    /// Everything outside it — inputs, callee-saved registers, the reload
    /// registers, the backend's own `fixed` scratch — is off limits by
    /// construction, and [`RegisterFile::checked`] proves the separation.
    pub scratch: RegSet,

    /// Fixed registers, outside the pool, that spilled operands reload into.
    ///
    /// Two suffice: every backend reads all of an instruction's sources before
    /// writing its destination, so a spilled destination register doubles as
    /// the temporary for one operand.
    pub reload: [Reg; 2],

    /// Registers the backend's own instruction emission clobbers, outside the
    /// allocator's knowledge: an ISA-level temp for a two-operand form, a
    /// gather's index register, and the like.
    ///
    /// The allocator never hands these out; declaring them is what lets
    /// [`RegisterFile::checked`] prove they miss the pool, the inputs and the
    /// reload pair — rather than a comment in an ISA file asserting it. Anything a backend takes for itself belongs here.
    pub fixed: &'static [Reg],

    /// How many registers this backend's encoding of `op` needs beyond the
    /// operands and destination — the instruction temps.
    ///
    /// A two-operand ISA needs one to break a destructive hazard; a sign-flip
    /// needs one to hold the mask. Those used to be `const`s outside the pool,
    /// reserved for the whole kernel because one instruction in it might want
    /// one. Declaring the demand here instead makes the temp an *allocated*
    /// value with a live range of exactly one instruction, so the register is
    /// the allocator's everywhere else.
    ///
    /// It lives on the register file because the file is already "the whole of
    /// what allocation needs to know about the target" — a backend that needs
    /// a temp is stating a fact about its register requirements, which is what
    /// this type is for.
    pub temps_for: fn(&ScheduledOp) -> u8,

    /// Bytes one register occupies when spilled — the backend's vector width.
    ///
    /// 16 for SSE2 and NEON, 32 for AVX2, 64 for AVX-512. This is the stride
    /// [`FrameLayout`](super::FrameLayout) lays spill slots out at, so every
    /// offset the emitter sees is already a real byte displacement. It was
    /// once a universal 16 that each wide backend divided back out at its
    /// every use site; a slot offset that failed to be a multiple of 16 would
    /// then have truncated two live values onto the same stack slot.
    pub vector_bytes: u32,
}

impl RegisterFile {
    /// Reject a register file whose regions overlap.
    ///
    /// Call it on every backend's `const` declaration: const evaluation runs
    /// the checks at build time, so an allocatable window that swallows a
    /// reload register cannot reach a running kernel.
    #[must_use]
    pub const fn checked(self) -> Self {
        assert!(
            self.scratch.len() >= Self::MIN_SCRATCH,
            "the allocatable pool is too small for the widest instruction's \
             operands, scratch and destination at once"
        );

        let mut i = 0;
        while i < self.inputs.len() {
            assert!(
                !self.scratch.contains(self.inputs[i]),
                "an input register is inside the allocatable pool: the \
                 allocator would hand a pre-colored register out twice"
            );
            i += 1;
        }

        let mut i = 0;
        while i < self.reload.len() {
            assert!(
                !self.scratch.contains(self.reload[i]),
                "a reload register is inside the allocatable pool: reloading \
                 a spilled operand would clobber another live value"
            );
            i += 1;
        }
        assert!(
            self.reload[0].0 != self.reload[1].0,
            "the two reload registers are the same register"
        );

        assert!(
            self.vector_bytes >= 16 && self.vector_bytes.is_power_of_two(),
            "vector_bytes must be a power of two of at least 16"
        );

        // Everything a backend reserves for its own emission must miss every
        // register the allocator reasons about. Without this the disjointness
        // lives only in comments.
        let mut i = 0;
        while i < self.fixed.len() {
            assert!(
                !self.scratch.contains(self.fixed[i]),
                "a fixed backend scratch register is inside the allocatable \
                 pool: emitting an instruction would clobber a live value"
            );
            let mut k = 0;
            while k < self.reload.len() {
                assert!(
                    self.fixed[i].0 != self.reload[k].0,
                    "a fixed backend scratch register aliases a reload register"
                );
                k += 1;
            }
            let mut k = 0;
            while k < self.inputs.len() {
                assert!(
                    self.fixed[i].0 != self.inputs[k].0,
                    "a fixed backend scratch register aliases a coordinate input"
                );
                k += 1;
            }
            i += 1;
        }

        self
    }

    /// The smallest pool any schedule can be allocated against.
    ///
    /// Every *value* survives a small pool by spilling, so the pool has no
    /// lower bound from values alone — that is why one register used to be an
    /// acceptable budget. Instruction scratch has no such escape: it is
    /// registers the encoder destroys mid-instruction, each of which must be a
    /// register, and one that is neither the destination nor any operand.
    ///
    /// Six, set by the widest single demand rather than by the widest op. A
    /// ternary is three operands plus one temp; AVX2's gather is one operand
    /// plus [`Scratch::MAX_TEMPS`] — four, since it assembles two 128-bit
    /// halves — and both leave room for the destination.
    ///
    /// Below this, shrinking the pool stops producing more spilling and starts
    /// producing an instruction with nowhere to put its scratch.
    pub const MIN_SCRATCH: u8 = Scratch::MAX_TEMPS as u8 + 2;

    /// Cap the scratch pool at a smaller budget, leaving every other region
    /// where it is.
    ///
    /// This is how [`EmitCtx::max_regs`](super::EmitCtx) forces spilling for
    /// pressure testing. It only ever *shrinks* the pool: a budget above the
    /// target's own count would hand the allocator registers this file has
    /// reserved for reloads or builtins. It does not shrink past
    /// [`MIN_SCRATCH`](Self::MIN_SCRATCH), which is not a budget question but
    /// an encoding one.
    #[must_use]
    pub const fn capped(self, max_scratch: Option<u8>) -> Self {
        match max_scratch {
            Some(n) => Self {
                scratch: self.scratch.take(if n < Self::MIN_SCRATCH {
                    Self::MIN_SCRATCH
                } else {
                    n
                }),
                ..self
            },
            None => self,
        }
    }

    /// This file as a scope *inside* a loop sees it: the pool minus every
    /// register carrying a value across that loop.
    ///
    /// Carried registers are not `fixed` — `fixed` is what a backend holds for
    /// its own encodings, and nothing does any more. These are ordinary
    /// allocations of an outer scope whose live range spans the scopes within,
    /// which is exactly what the nest's liveness says and what allocating each
    /// region against the full pool used to ignore.
    #[must_use]
    pub const fn inside(self, carried: RegSet) -> Self {
        Self {
            scratch: self.scratch.without(carried),
            ..self
        }
    }

    /// The allocatable scratch registers, low to high.
    fn scratch(&self) -> impl Iterator<Item = Reg> + use<> {
        self.scratch.iter()
    }
}

/// One scope of a loop nest: the enclosing regions, outermost first, and then
/// the innermost body.
///
/// The derived order is execution order on the first pass through the nest —
/// `Region(0)` runs once per call, the last region once per iteration of the
/// next-to-innermost binder, `Body` at every sample. That order is all a
/// [`Point`] needs to compare, and it is deliberately no more than that: it is
/// **not** a trip-count model and must not be read as one.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    /// An enclosing region; `Region(0)` is the outermost.
    Region(usize),
    /// The innermost body, inside every back edge.
    Body,
}

/// A program point in a loop nest: a scope, and a position in that scope's
/// schedule. Ordered lexicographically.
///
/// Program point *is* (scope, index into that scope's schedule), which is why
/// this is the coordinate a placement's ranges are stated in.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Point {
    /// Which scope of the nest.
    pub scope: Scope,
    /// Position in that scope's schedule.
    pub index: usize,
}

impl Point {
    /// The last point of the nest.
    ///
    /// Every scope's subtree ends inside the body, so there is exactly one of
    /// these — which is what makes "live to the tail" a single, comparable
    /// answer for a value read across any back edge.
    pub const TAIL: Self = Self {
        scope: Scope::Body,
        index: usize::MAX,
    };
}

/// Where the allocator decided a value lives, over one range of its life.
///
/// Deliberately carries no stack address: choosing that a value spills and
/// choosing *where* it spills are different decisions, and the second belongs
/// to [`FrameLayout`](super::FrameLayout), which is what knows about frames.
/// The emitter reads the composition of the two as [`Loc`](super::Loc).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Where {
    /// In this register.
    Reg(Reg),
    /// Evicted to a stack slot.
    Spilled,
    /// Evicted, but it is a constant (these are the `f32` bits): it lives
    /// nowhere and is re-emitted at each use, which beats a store plus a
    /// reload.
    Remat(u32),
}

/// One range of a value's life: from `from` (inclusive) until the next range's
/// `from` (exclusive), the value lives at `at`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Span {
    /// The program point this range starts at.
    pub from: Point,
    /// Where the value lives over it.
    pub at: Where,
}

/// Where a value lives, at every point in the nest.
///
/// A **non-empty, strictly increasing sequence** of [`Span`]s. Non-empty by
/// construction — a value lives somewhere from its definition on — which is
/// why the first range is a field rather than the head of a `Vec` something
/// could empty; the rest is usually empty, and an empty `Vec` does not
/// allocate.
///
/// One location for a whole life was the old shape, and it is the shape that
/// makes two things unsayable. A value hot in part of a region and cold in the
/// rest cannot hold a register for the hot part only; and a root computed in
/// an outer region and read inside the loops within cannot be in that region's
/// register *and* in a slot for the loops — which is two locations over one
/// life, and is what the nest-wide map here has to express. That second one is
/// why the `carries` side-channel is gone: it was the half of this answer the
/// old shape could not hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    first: Span,
    rest: Vec<Span>,
}

impl Placement {
    /// A placement that starts at `first` and never changes.
    #[must_use]
    pub fn new(first: Span) -> Self {
        Self {
            first,
            rest: Vec::new(),
        }
    }

    /// The same placement, changing to `next.at` from `next.from` onward.
    ///
    /// # Panics
    /// In debug builds, if `next` does not start strictly after every range
    /// already here — the sequence is increasing, and an out-of-order range
    /// would make [`Placement::at`] answer with a location the value had
    /// already left.
    #[must_use]
    pub fn then(mut self, next: Span) -> Self {
        debug_assert!(
            next.from > self.rest.last().unwrap_or(&self.first).from,
            "placement ranges must strictly increase"
        );
        self.rest.push(next);
        self
    }

    /// The point this value is defined at — where its first range starts.
    #[must_use]
    pub fn defined_at(&self) -> Point {
        self.first.from
    }

    /// Where the value lives at `point`.
    ///
    /// Total: the last range whose `from` is at or before `point`. A query
    /// before the definition — which no caller can make, since an operand is
    /// read after it is defined — answers with the first range rather than
    /// panicking.
    #[must_use]
    pub fn at(&self, point: Point) -> Where {
        let after = self.rest.partition_point(|s| s.from <= point);
        match after.checked_sub(1) {
            Some(i) => self.rest[i].at,
            None => self.first.at,
        }
    }

    /// Every range of this value's life, in order.
    pub fn spans(&self) -> impl Iterator<Item = Span> + use<'_> {
        core::iter::once(self.first).chain(self.rest.iter().copied())
    }

    /// Every location this value occupies, in order.
    pub fn locations(&self) -> impl Iterator<Item = Where> + use<'_> {
        self.spans().map(|s| s.at)
    }

    /// Whether any range of this value's life is in a stack slot.
    #[must_use]
    pub fn spills(&self) -> bool {
        self.locations().any(|at| at == Where::Spilled)
    }

    /// Every register this value occupies over its life.
    pub fn registers(&self) -> impl Iterator<Item = Reg> + use<'_> {
        self.locations().filter_map(|at| match at {
            Where::Reg(r) => Some(r),
            Where::Spilled | Where::Remat(_) => None,
        })
    }
}

/// One scope's evaluation order and instruction scratch — everything about a
/// scope that is not a placement.
///
/// The schedule is an *output* because choosing it is part of allocating.
/// [`LinearScan`] hands back the order it was given; an allocator whose
/// register assignment falls out of evaluation order — Sethi-Ullman, where the
/// heavier subtree is emitted first and the register is a function of tree
/// position — hands back the order it chose.
#[derive(Debug)]
struct ScopeCode {
    /// Evaluation order: the schedule the emitter walks.
    schedule: Vec<Def>,
    /// The scratch each position in `schedule` may destroy.
    ///
    /// Indexed by schedule position rather than by value because scratch is not
    /// a value: it holds nothing before the instruction and nothing after, so
    /// it has no `ValueId` and no place in the placements.
    scratch: Vec<Scratch>,
    /// Values this scope computes for the scopes inside it, in slot order.
    /// Empty for the body, which parks nothing.
    roots: Vec<ValueId>,
}

/// The registers one instruction may destroy for its own duration.
///
/// Each field is a role, not a slot — the allocator picks a register for each
/// one it fills, disjoint from the instruction's operands, its destination, and
/// each other. Reading them positionally out of a shared array is exactly the
/// convention this replaced.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Scratch {
    /// The encoding's own scratch: a sign mask, a Newton-Raphson correction,
    /// the halves a gather assembles its result from — as many registers as
    /// [`RegisterFile::temps_for`] asked for, and no more.
    ///
    /// Private, and read through [`Scratch::temp`], because the *order* here is
    /// a contract between one backend's `temps_for` and that same backend's
    /// encoder. That is a local agreement inside one ISA file; letting anything
    /// else index it would make it a convention spanning the codebase, which is
    /// what the roles below exist to avoid.
    temps: [Option<Reg>; Scratch::MAX_TEMPS],

    /// A third register to reload an operand into.
    ///
    /// `Select` is the one instruction that can need three reload targets at
    /// once. Its mask has to end up in `dst`, so a spilled mask consumes the
    /// destination, and the two arms then need a target each; [`RegisterFile`]
    /// carries two. The third used to be `select_reload`, a register held out
    /// of *every* kernel's pool for the sake of the rare kernel that reaches
    /// this.
    ///
    /// Reserved for every `Select` rather than only the ones that turn out to
    /// need it, because a value resident when its reader is allocated can still
    /// be evicted by a later instruction — `Placement` is one answer per value
    /// for its whole life — so residency read at this point is not yet final.
    /// Over-reserving costs one register at a `Select`; the register it
    /// replaces cost one everywhere.
    pub arm_reload: Option<Reg>,
}

impl Scratch {
    /// The most scratch registers any one encoding asks for.
    ///
    /// Four: AVX2 assembles a 256-bit gather from two 128-bit halves, which
    /// costs the half-sequence's own index and value registers plus one of
    /// each to carry the high half while the low one is built.
    pub const MAX_TEMPS: usize = 4;

    /// A `Scratch` with the registers a test wants to hand an encoder.
    ///
    /// The allocator is what fills these in production; a test that exercises
    /// one encoding in isolation has no allocator, so it says outright which
    /// registers the encoder may destroy.
    #[cfg(test)]
    #[must_use]
    pub const fn for_test(temps: Option<[Reg; Self::MAX_TEMPS]>, arm_reload: Option<Reg>) -> Self {
        let temps = match temps {
            Some([a, b, c, d]) => [Some(a), Some(b), Some(c), Some(d)],
            None => [None; Self::MAX_TEMPS],
        };
        Self { temps, arm_reload }
    }

    /// The `i`'th register this instruction's encoding asked for.
    ///
    /// `i` is the backend's own numbering, matching the count its
    /// [`RegisterFile::temps_for`] returned.
    #[must_use]
    pub fn temp(&self, i: usize) -> Option<Reg> {
        self.temps.get(i).copied().flatten()
    }
}

/// What an allocator makes of a [`ScopedSchedule`]: **one** answer to one
/// question — where does each value live, at every point in the nest.
///
/// The per-region placement maps and the `carries` side-channel were two
/// encodings of that answer, and one of them existed because the other could
/// not say it: a root computed in a region and read by the loops inside lives
/// in that region's register and then in a slot, which is two locations over
/// one life. A ranged [`Placement`] says that directly, so there is one map
/// here and nothing beside it.
///
/// `ValueId`s are *not* partitioned by the nest — a `Var` or `Const` leaf
/// feeding both an invariant expression and a varying one appears in both
/// scopes' schedules, with an independently chosen location in each. That is
/// another thing a sequence says and a single answer per value cannot.
#[derive(Debug)]
pub struct NestAllocation {
    /// Dense by `ValueId.0`, over the whole nest.
    placements: Vec<Option<Placement>>,
    /// One per input region, in the same order — outermost first.
    regions: Vec<ScopeCode>,
    /// The innermost body.
    body: ScopeCode,
}

impl NestAllocation {
    /// How many enclosing regions the nest has.
    #[must_use]
    pub fn regions(&self) -> usize {
        self.regions.len()
    }

    /// The allocation as `scope` reads it.
    ///
    /// # Panics
    /// If `scope` names a region this nest does not have.
    #[must_use]
    pub fn scope(&self, scope: Scope) -> Allocation<'_> {
        assert!(
            self.code(scope).is_some(),
            "{scope:?} is not a scope of this nest"
        );
        Allocation { nest: self, scope }
    }

    /// The innermost body — the whole answer for a loop-free schedule.
    #[must_use]
    pub fn body(&self) -> Allocation<'_> {
        self.scope(Scope::Body)
    }

    fn code(&self, scope: Scope) -> Option<&ScopeCode> {
        match scope {
            Scope::Region(i) => self.regions.get(i),
            Scope::Body => Some(&self.body),
        }
    }

    /// Where `v` lives, at every point in the nest.
    ///
    /// # Panics
    /// If `v` is in no scope of this nest.
    #[must_use]
    pub fn placement(&self, v: ValueId) -> &Placement {
        self.placements
            .get(v.0 as usize)
            .and_then(Option::as_ref)
            .unwrap_or_else(|| panic!("{v:?} is not in this allocation"))
    }

    /// Where `v` lives at program point `at`.
    ///
    /// # Panics
    /// If `v` is in no scope of this nest.
    #[must_use]
    pub fn where_at(&self, v: ValueId, at: Point) -> Where {
        self.placement(v).at(at)
    }

    /// The register a root is **carried** in across the loops inside the
    /// region that computes it, if it is carried at all.
    ///
    /// A root is carried exactly when its placement inside the loops is a
    /// register: the body reads it from there on every iteration instead of
    /// reloading it from a slot at every use. There is no separate map saying
    /// so — that map was `carries`, and the placement already answers.
    ///
    /// # Panics
    /// If `root` is in no scope of this nest.
    #[must_use]
    pub fn carried(&self, root: ValueId) -> Option<Reg> {
        match self.where_at(
            root,
            Point {
                scope: Scope::Body,
                index: 0,
            },
        ) {
            Where::Reg(r) => Some(r),
            Where::Spilled | Where::Remat(_) => None,
        }
    }

    /// Override where a value lives, for the whole of its life.
    ///
    /// The emitter's own tests pin a value somewhere the allocator did not
    /// choose. One write, so the placement cannot desync from itself.
    ///
    /// # Panics
    /// If `v` is in no scope of this nest — a placement has to start
    /// somewhere, and only the allocation knows where `v` is defined.
    pub fn place(&mut self, v: ValueId, at: Where) {
        let from = self.placement(v).defined_at();
        self.placements[v.0 as usize] = Some(Placement::new(Span { from, at }));
    }
}

/// The allocation as one scope reads it: that scope's schedule and scratch,
/// and the nest-wide placements resolved at points inside it.
///
/// The scope is baked in, so callers hand over a *local* schedule index and
/// cannot name a point in some other scope by accident.
#[derive(Copy, Clone, Debug)]
pub struct Allocation<'a> {
    nest: &'a NestAllocation,
    scope: Scope,
}

impl<'a> Allocation<'a> {
    /// Evaluation order: the schedule the emitter walks.
    #[must_use]
    pub fn schedule(&self) -> &'a [Def] {
        &self.code().schedule
    }

    /// The values this scope computes for the scopes inside it, in slot order.
    #[must_use]
    pub fn roots(&self) -> &'a [ValueId] {
        &self.code().roots
    }

    /// The scratch the instruction at schedule position `i` may destroy.
    #[must_use]
    pub fn scratch(&self, i: usize) -> Scratch {
        self.code().scratch.get(i).copied().unwrap_or_default()
    }

    /// Where `v` lives at position `index` of this scope.
    ///
    /// The query carries its own point because a placement is a schedule, not
    /// an annotation: asking where a value is without saying *when* is a
    /// question with no answer once a live range can be split.
    ///
    /// # Panics
    /// If `v` is in no scope of this nest.
    #[must_use]
    pub fn where_at(&self, v: ValueId, index: usize) -> Where {
        self.nest.where_at(v, self.point(index))
    }

    /// The register a root is carried in across the loops inside its region.
    #[must_use]
    pub fn carried(&self, root: ValueId) -> Option<Reg> {
        self.nest.carried(root)
    }

    /// Where `v` lives at every point in the nest.
    ///
    /// # Panics
    /// If `v` is in no scope of this nest.
    #[must_use]
    pub fn placement(&self, v: ValueId) -> &'a Placement {
        self.nest.placement(v)
    }

    /// The points inside *this* scope at which `v` changes place, in order.
    ///
    /// A placement is nest-wide; emitting one scope needs the part of it that
    /// happens here. This is what lets the emitter maintain its location table
    /// incrementally — one pass, O(total ranges) — rather than asking where
    /// every value is at every instruction.
    ///
    /// # Panics
    /// If `v` is in no scope of this nest.
    pub fn transitions(self, v: ValueId) -> impl Iterator<Item = (usize, Where)> + use<'a> {
        let scope = self.scope;
        self.nest
            .placement(v)
            .spans()
            .filter(move |s| s.from.scope == scope)
            .map(|s| (s.from.index, s.at))
    }

    /// The first program point *inside* this scope: where the loops it
    /// encloses begin, and where a root it parks is picked up.
    ///
    /// The body encloses nothing, so its own end is the answer there — a point
    /// no range starts at, which is exactly "there is nothing inside".
    #[must_use]
    pub fn inner_head(&self) -> Point {
        match self.scope {
            Scope::Region(i) if i + 1 < self.nest.regions.len() => Point {
                scope: Scope::Region(i + 1),
                index: 0,
            },
            Scope::Region(_) => Point {
                scope: Scope::Body,
                index: 0,
            },
            Scope::Body => Point::TAIL,
        }
    }

    /// Whether this scope *reads* `v` from an enclosing region's park rather
    /// than computing it.
    ///
    /// Such a value's entry in this schedule is a placeholder, and its address
    /// is a hoist slot that outlives every region's own frame — so it is not
    /// this frame's to place. Narrower than "defined elsewhere": a `Const`
    /// leaf shared with an enclosing region is genuinely computed here too,
    /// and does need a location of its own.
    #[must_use]
    pub fn parked_by_an_enclosing_scope(&self, v: ValueId) -> bool {
        let outside = match self.scope {
            Scope::Region(i) => i,
            Scope::Body => self.nest.regions.len(),
        };
        self.nest.regions[..outside]
            .iter()
            .any(|r| r.roots.contains(&v))
    }

    fn point(&self, index: usize) -> Point {
        Point {
            scope: self.scope,
            index,
        }
    }

    fn code(&self) -> &'a ScopeCode {
        self.nest
            .code(self.scope)
            .unwrap_or_else(|| unreachable!("`NestAllocation::scope` checked this"))
    }
}

/// Assign physical registers to an expression DAG.
///
/// A pure function from a program and a register file to a placement for every
/// value in it. Purity is load-bearing, not incidental: the collapse-loop
/// driver runs allocation once to size a stack frame and again to emit into
/// that frame, and a disagreement between the two runs misplaces every spill.
///
/// Allocation is not a local decision. Liveness needs the whole program, and
/// the eviction rule that makes the difference — Belady's, evict whatever is
/// used farthest in the future — is *defined* in terms of the future. So an
/// implementation sees the entire DAG, and owes an answer for every value in
/// the schedule it returns.
///
/// Running out of registers is not a failure; it is a spill. There is no error
/// case: a DAG this cannot allocate is a DAG the pipeline should never have
/// produced, and it panics at the point of failure rather than handing a
/// caller a string it can only propagate.
pub trait RegisterAllocator {
    /// Place every value in a loop nest, and choose the evaluation order.
    ///
    /// This is the whole job, and it is deliberately the *only* required
    /// method. The nest — not a flat schedule — is the honest input, because
    /// where a value is read decides what keeping it in a register is worth:
    /// a read inside a loop costs its reload once per iteration, a read in the
    /// prologue costs it once. An allocator handed a flat `Vec<Def>` cannot
    /// tell those apart, so it cannot price them, and the only policy it can
    /// implement is one that ignores the difference.
    ///
    /// Taking the nest by value because choosing the order is part of the job:
    /// an implementation may permute what it is handed, and returns the order
    /// it settled on.
    fn allocate_nest(&self, nest: ScopedSchedule, file: &RegisterFile) -> NestAllocation;

    /// A loop-free schedule, which is the degenerate nest: one body, no
    /// regions, nothing carried across anything.
    ///
    /// Provided rather than required, and in that direction on purpose. It
    /// used to be the other way round — `allocate` required, `allocate_nest`
    /// defaulted to calling it once per region — which put the loop policy in
    /// this trait's default body, where every implementation inherited it and
    /// none of them owned it. A trait should say what an allocator answers,
    /// not how.
    fn allocate(&self, dag: Vec<Def>, file: &RegisterFile) -> NestAllocation {
        self.allocate_nest(
            ScopedSchedule {
                regions: Vec::new(),
                body: dag,
            },
            file,
        )
    }
}

/// A schedule split by scope: the innermost body, plus one region per binder
/// it can be lifted out of.
///
/// This is the loop nest as data. `regions` runs outermost first, so
/// `regions[0]` is what happens once per call and the last region is what
/// happens once per iteration of the next-to-innermost binder; `body` is what
/// happens at every sample. Each region's `roots` are the values it computes
/// for the regions inside it.
///
/// The body is a field rather than the last element of `regions` because it
/// is genuinely a different thing: it is inside every back edge, it produces
/// the result, and it parks nothing. A list that could hold it would let the
/// two be confused.
pub struct ScopedSchedule {
    /// One per binder, outermost first. A binder nothing can be lifted out of
    /// still gets a region; it is simply empty.
    pub regions: Vec<ScopeRegion>,
    /// The innermost region: evaluated at every sample.
    pub body: Vec<Def>,
}

/// One scope of a [`ScopedSchedule`].
pub struct ScopeRegion {
    /// Values this region computes for the ones inside it.
    pub roots: Vec<ValueId>,
    /// What it computes, in topological order.
    pub schedule: Vec<Def>,
}

/// Linear scan with Belady eviction and constant rematerialization.
///
/// Two passes. The first walks the program computing each value's last use;
/// the second walks it forward assigning registers, and at each definition:
/// 1. Frees registers whose values have expired.
/// 2. Takes a free scratch register if there is one.
/// 3. Otherwise evicts — preferring a constant (free to rematerialize, no
///    memory traffic) and otherwise the live value whose next use is farthest
///    in the future.
///
/// The evaluation order it returns is the one it was given: the arena's
/// append-only structure already guarantees a topological order, so there is
/// nothing to linearize.
///
/// Coordinate inputs are pinned to `file.inputs` and never enter the scratch
/// pool, which [`RegisterFile::checked`] keeps disjoint from them.
///
/// O(n × k) for n values and a k-register pool. For the pools here (4–10
/// registers) that is effectively O(n).
///
/// The DAGs reaching this are in SSA form, which makes their interference
/// graphs chordal — the shape on which greedy coloring is optimal. A
/// graph-coloring allocator therefore lived beside this one, unused, for as
/// long as this file existed: it never learned to rematerialize constants, and
/// on a pool this small that is the difference that decides the generated code.
#[derive(Copy, Clone, Debug, Default)]
pub struct LinearScan;

impl RegisterAllocator for LinearScan {
    fn allocate_nest(&self, nest: ScopedSchedule, file: &RegisterFile) -> NestAllocation {
        // Outermost first, because that is the direction liveness flows: a
        // value a region computes for the scopes inside it is live across
        // every iteration of every loop between here and its last use. A
        // register holding it is therefore unavailable to all of them, which
        // is what allocating each region against the full pool used to miss.
        let mut carried = RegSet::EMPTY;
        let mut placements: Vec<Option<Placement>> = Vec::new();
        // Roots of the regions already handled: values an inner scope reads
        // from a park rather than computing, so its scan's answer for them is
        // a placeholder's and must not overwrite the park.
        let mut parked: BTreeSet<ValueId> = BTreeSet::new();
        let mut regions: Vec<ScopeCode> = Vec::with_capacity(nest.regions.len());
        let region_count = nest.regions.len();

        // Uses in the innermost body are what a carry actually saves: one
        // reload per use, per iteration. Counted once, up front.
        let mut body_uses: BTreeMap<ValueId, usize> = BTreeMap::new();
        for def in &nest.body {
            for operand in operands(&def.op) {
                *body_uses.entry(operand).or_insert(0) += 1;
            }
        }

        for (index, region) in nest.regions.into_iter().enumerate() {
            let scope = Scope::Region(index);
            let scoped = file.inside(carried);
            let scan = self.scan(region.schedule, &scoped);

            // Carry the roots the body reads most, while the body keeps a
            // pool it can still allocate in. Every carry costs one register
            // for the whole loop and saves one reload per use per iteration,
            // so the ordering is by use count and the cap is the floor.
            let budget = file
                .scratch
                .len()
                .saturating_sub(carried.len())
                .saturating_sub(RegisterFile::MIN_SCRATCH);
            let mut ranked: Vec<(usize, ValueId)> = region
                .roots
                .iter()
                .map(|v| (body_uses.get(v).copied().unwrap_or(0), *v))
                .filter(|(uses, _)| *uses > 0)
                .collect();
            // Descending by use count, then by id so the choice is
            // deterministic — two roots read the same number of times must
            // not depend on map iteration order.
            ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.0.cmp(&b.1.0)));

            // Everything the region's own code touches: placed values AND
            // per-instruction scratch. A temp is a pool register that no
            // placement records, so taking the complement of the locations
            // alone would hand out a register the prologue destroys.
            let mut used: Vec<Reg> = scan
                .locations
                .iter()
                .flatten()
                .filter_map(|at| match at {
                    Where::Reg(r) => Some(*r),
                    Where::Spilled | Where::Remat(_) => None,
                })
                .collect();
            for scratch in &scan.scratch {
                used.extend(scratch.temps.iter().flatten().copied());
                used.extend(scratch.arm_reload);
            }
            let free = scoped.scratch.without(RegSet::of(&used));
            let mut available = free.iter();
            let mut carries: BTreeMap<ValueId, Reg> = BTreeMap::new();
            for (_, vid) in ranked.into_iter().take(budget as usize) {
                let Some(reg) = available.next() else { break };
                carried = carried.union(RegSet::of(&[reg]));
                carries.insert(vid, reg);
            }

            record(&mut placements, scope, &scan, &parked);

            // A root outlives the region computing it, so its placement has a
            // second range: from the first point of the scope inside, it is
            // either the register carrying it across the loops or the slot it
            // was parked in. Two ranges over one life — the thing a single
            // location per value could not say, and the reason the emitter
            // needed a `carries` map beside it.
            let inside = if index + 1 < region_count {
                Scope::Region(index + 1)
            } else {
                Scope::Body
            };
            for root in &region.roots {
                let park = Span {
                    from: Point {
                        scope: inside,
                        index: 0,
                    },
                    at: match carries.get(root) {
                        Some(reg) => Where::Reg(*reg),
                        None => Where::Spilled,
                    },
                };
                let in_region = placements[root.0 as usize]
                    .take()
                    .unwrap_or_else(|| unreachable!("a region computes its own roots"));
                placements[root.0 as usize] = Some(in_region.then(park));
            }

            parked.extend(region.roots.iter().copied());
            regions.push(ScopeCode {
                schedule: scan.schedule,
                scratch: scan.scratch,
                roots: region.roots,
            });
        }

        let body = self.scan(nest.body, &file.inside(carried));
        record(&mut placements, Scope::Body, &body, &parked);

        NestAllocation {
            placements,
            regions,
            body: ScopeCode {
                schedule: body.schedule,
                scratch: body.scratch,
                roots: Vec::new(),
            },
        }
    }
}

/// Fold one scope's scan into the nest-wide map.
///
/// A `parked` value is skipped: its entry in this schedule is a placeholder
/// the emitter never emits, and the region that computes it has already said
/// where it lives from here on. Everything else this scope schedules gets a
/// range at its own definition — including a `Var`/`Const` leaf an enclosing
/// scope also computes, which is genuinely rebuilt here and genuinely may land
/// somewhere else.
fn record(
    placements: &mut Vec<Option<Placement>>,
    scope: Scope,
    scan: &Scan,
    parked: &BTreeSet<ValueId>,
) {
    if placements.len() < scan.locations.len() {
        placements.resize(scan.locations.len(), None);
    }
    for (index, def) in scan.schedule.iter().enumerate() {
        if parked.contains(&def.value) {
            continue;
        }
        let Some(at) = scan.locations[def.value.0 as usize] else {
            continue;
        };
        let from = Point { scope, index };
        let slot = &mut placements[def.value.0 as usize];
        *slot = Some(match slot.take() {
            // A leaf an enclosing scope also computes: another range, since
            // the two scans chose independently.
            Some(prior) if prior.at(from) != at => prior.then(Span { from, at }),
            Some(prior) => prior,
            None => Placement::new(Span { from, at }),
        });
    }
}

/// One scope, scanned straight through: a location per value and the scratch
/// per instruction.
///
/// A straight-line scan has no ranges to state — every value it places, it
/// places for the whole of the scope it is scanning. Turning those into
/// nest-wide [`Placement`]s is [`LinearScan::allocate_nest`]'s job, because
/// only the nest knows what happens to a value after this scope ends.
struct Scan {
    schedule: Vec<Def>,
    /// Dense by `ValueId.0`, over this scope alone.
    locations: Vec<Option<Where>>,
    scratch: Vec<Scratch>,
}

impl LinearScan {
    /// One region, scanned straight through.
    fn scan(&self, dag: Vec<Def>, file: &RegisterFile) -> Scan {
        let vec_len = dag
            .iter()
            .map(|def| def.value.0 as usize + 1)
            .max()
            .unwrap_or(0);
        let mut placements: Vec<Option<Where>> = vec![None; vec_len];

        let mut scratch_for: Vec<Scratch> = vec![Scratch::default(); dag.len()];

        if dag.is_empty() {
            return Scan {
                schedule: dag,
                locations: placements,
                scratch: scratch_for,
            };
        }

        // Pass one: last use of each value — the latest schedule index of any
        // operation that reads it. A value nothing reads (the root) is last
        // used at its own definition.
        let mut last_use: Vec<usize> = vec![usize::MAX; vec_len];
        for (i, def) in dag.iter().enumerate() {
            if last_use[def.value.0 as usize] == usize::MAX {
                last_use[def.value.0 as usize] = i;
            }
        }
        for (i, def) in dag.iter().enumerate() {
            for operand in operands(&def.op) {
                let lu = &mut last_use[operand.0 as usize];
                *lu = (*lu).max(i);
            }
        }

        // A constant is never worth a spill slot: bringing it back is one
        // instruction either way.
        let mut const_bits: Vec<Option<u32>> = vec![None; vec_len];
        for def in &dag {
            if let ScheduledOp::Const(val) = def.op {
                const_bits[def.value.0 as usize] = Some(val.to_bits());
            }
        }

        // Coordinate inputs are pinned to the registers the ABI delivers them
        // in. The scratch pool excludes those registers, so a pinned value
        // never competes for one.
        for def in &dag {
            if let ScheduledOp::Var(i) = def.op {
                placements[def.value.0 as usize] = Some(Where::Reg(input_register(file, i)));
            }
        }

        // reg_owner[i] = the value currently held in the i'th scratch register.
        let mut reg_owner: Vec<Option<ValueId>> = vec![None; file.scratch.len() as usize];
        let scratch: Vec<Reg> = file.scratch().collect();

        // Pass two: forward over the program in evaluation order.
        for (i, def) in dag.iter().enumerate() {
            let vid = &def.value;

            for slot in reg_owner.iter_mut() {
                if let Some(owner) = *slot
                    && last_use[owner.0 as usize] < i
                {
                    *slot = None;
                }
            }

            // Scratch, reserved before the destination is placed: the encoder
            // writes it while every operand is still live and before the
            // destination is written, so it may share a register with neither.
            let mut taken: [Option<usize>; Scratch::MAX_TEMPS + 1] = [None; Scratch::MAX_TEMPS + 1];
            let mut reserve = |role: usize,
                               reg_owner: &mut Vec<Option<ValueId>>,
                               placements: &mut Vec<Option<Where>>|
             -> Reg {
                let slot = reserve_scratch(def, reg_owner, &last_use, &const_bits, &taken);
                taken[role] = Some(slot);
                // Whatever was there loses its register for good: `placements`
                // is one answer per value, so an evicted value is spilled over
                // its whole life, exactly as destination eviction below does.
                if let Some(evicted) = reg_owner[slot].take() {
                    placements[evicted.0 as usize] = Some(match const_bits[evicted.0 as usize] {
                        Some(bits) => Where::Remat(bits),
                        None => Where::Spilled,
                    });
                }
                scratch[slot]
            };

            let wanted = (file.temps_for)(&def.op) as usize;
            assert!(
                wanted <= Scratch::MAX_TEMPS,
                "a backend asked for {wanted} scratch registers for one \
                 instruction; `Scratch::MAX_TEMPS` is {}",
                Scratch::MAX_TEMPS
            );
            for role in 0..wanted {
                scratch_for[i].temps[role] = Some(reserve(role, &mut reg_owner, &mut placements));
            }
            if matches!(def.op, ScheduledOp::Ternary(OpKind::Select, ..)) {
                scratch_for[i].arm_reload =
                    Some(reserve(Scratch::MAX_TEMPS, &mut reg_owner, &mut placements));
            }
            let reserved = taken;

            if placements[vid.0 as usize].is_some() {
                // Pre-colored. The temp's slot stays free for the next
                // instruction: nothing owns it after this one.
                continue;
            }

            if let Some(free) = reg_owner
                .iter()
                .enumerate()
                .position(|(idx, slot)| slot.is_none() && !reserved.contains(&Some(idx)))
            {
                placements[vid.0 as usize] = Some(Where::Reg(scratch[free]));
                reg_owner[free] = Some(*vid);
                continue;
            }

            // Nothing free — evict. A constant goes first whatever its next
            // use, because rematerializing it costs no memory traffic; among
            // non-constants, Belady: whoever is used farthest out.
            let mut best_const: Option<(usize, usize)> = None; // (slot, last use)
            let mut best_any: (usize, usize) = (0, 0);

            for (slot_idx, slot) in reg_owner.iter().enumerate() {
                if reserved.contains(&Some(slot_idx)) {
                    continue; // Reserved for this instruction's own scratch.
                }
                let Some(owner) = *slot else { continue };
                let lu = last_use[owner.0 as usize];
                if const_bits[owner.0 as usize].is_some()
                    && best_const.is_none_or(|(_, best)| lu > best)
                {
                    best_const = Some((slot_idx, lu));
                }
                if lu > best_any.1 {
                    best_any = (slot_idx, lu);
                }
            }

            let evict_slot = best_const.map_or(best_any.0, |(slot, _)| slot);
            debug_assert!(
                !reserved.contains(&Some(evict_slot)),
                "the destination must not evict this instruction's own scratch"
            );
            let occupant = reg_owner[evict_slot].expect("all slots occupied but none found");

            // Which of the two goes to memory: a constant always loses, and
            // otherwise Belady decides.
            let evict_new = if const_bits[vid.0 as usize].is_some() {
                true
            } else if const_bits[occupant.0 as usize].is_some() {
                false
            } else {
                last_use[vid.0 as usize] >= last_use[occupant.0 as usize]
            };

            let loser = if evict_new { *vid } else { occupant };
            placements[loser.0 as usize] = Some(match const_bits[loser.0 as usize] {
                Some(bits) => Where::Remat(bits),
                None => Where::Spilled,
            });

            if !evict_new {
                placements[vid.0 as usize] = Some(Where::Reg(scratch[evict_slot]));
                reg_owner[evict_slot] = Some(*vid);
            }
        }

        Scan {
            schedule: dag,
            locations: placements,
            scratch: scratch_for,
        }
    }
}

/// The register a `Var(i)` is pinned to.
///
/// A `Var` past the coordinate inputs is not a kernel the target cannot run;
/// it is a `Var` that should have been lowered away (a reduce binder, a
/// manifold-param slot) or an axis that does not exist. Both are bugs upstream
/// of here, and neither is something a caller could act on — so it fails at
/// the point of failure, with a stack trace, rather than as a string three
/// frames up.
#[cold]
#[inline(never)]
fn var_out_of_range(i: u8, inputs: usize) -> ! {
    panic!(
        "Var({i}) names coordinate {i} but the target supplies {inputs} \
         (X, Y, Z, W) — `passes::legalize` lowers every other Var, so this is \
         a bypassed pipeline or a missing lowering, not a bad kernel"
    )
}

fn input_register(file: &RegisterFile, i: u8) -> Reg {
    match file.inputs.get(i as usize) {
        Some(&reg) => reg,
        None => var_out_of_range(i, file.inputs.len()),
    }
}

/// The values an operation reads.
/// A backend whose encodings never need a register beyond their operands.
///
/// The default for [`RegisterFile::temps_for`]; naming it keeps the field
/// total, so a new backend states its answer rather than inheriting one.
#[must_use]
pub fn no_temps(_op: &ScheduledOp) -> u8 {
    0
}

/// Pick the scratch slot the instruction at `i` may destroy as its temp.
///
/// A free slot if there is one; otherwise the same eviction rule the
/// destination uses — a constant first, since rematerializing it costs no
/// memory traffic, and Belady among the rest.
///
/// Operands are excluded outright: the encoder writes the temp *before* it
/// reads the operands (that is what the temp is for), so sharing a register
/// with one would feed the instruction its own scratch. So are the roles this
/// instruction has already reserved, which is what `taken` carries.
///
/// A slot always survives both exclusions, and
/// [`RegisterFile::MIN_SCRATCH`] is what makes that true: the instructions
/// wanting several temps are the gathers, which read one operand, and the
/// ones reading three operands want at most one temp plus an arm reload — so
/// four exclusions is the worst case against a floor of `MAX_TEMPS + 2`.
fn reserve_scratch(
    def: &Def,
    reg_owner: &[Option<ValueId>],
    last_use: &[usize],
    const_bits: &[Option<u32>],
    taken: &[Option<usize>],
) -> usize {
    let is_operand = |v: ValueId| operands(&def.op).any(|o| o == v);

    let mut best_const: Option<(usize, usize)> = None;
    let mut best_any: Option<(usize, usize)> = None;
    for (slot_idx, slot) in reg_owner.iter().enumerate() {
        if taken.contains(&Some(slot_idx)) {
            continue; // Already this instruction's, in another role.
        }
        let Some(owner) = *slot else {
            return slot_idx; // Free: nothing to evict.
        };
        if is_operand(owner) {
            continue;
        }
        let lu = last_use[owner.0 as usize];
        if const_bits[owner.0 as usize].is_some() && best_const.is_none_or(|(_, best)| lu > best) {
            best_const = Some((slot_idx, lu));
        }
        if best_any.is_none_or(|(_, best)| lu > best) {
            best_any = Some((slot_idx, lu));
        }
    }

    best_const
        .or(best_any)
        .map(|(slot, _)| slot)
        .unwrap_or_else(|| {
            unreachable!(
                "every scratch register holds an operand of one instruction, \
                 so the pool is at most three registers — `RegisterFile::checked` \
                 and `capped` both hold it at {}",
                RegisterFile::MIN_SCRATCH
            )
        })
}

fn operands(sop: &ScheduledOp) -> impl Iterator<Item = ValueId> + use<'_> {
    let (a, b, c) = match sop {
        ScheduledOp::Var(_) | ScheduledOp::Const(_) => (None, None, None),
        ScheduledOp::Unary(_, a) | ScheduledOp::ShiftImm(_, a, _) | ScheduledOp::Gather(a, _) => {
            (Some(*a), None, None)
        }
        ScheduledOp::Binary(_, a, b) => (Some(*a), Some(*b), None),
        ScheduledOp::Ternary(_, a, b, c) => (Some(*a), Some(*b), Some(*c)),
    };
    [a, b, c].into_iter().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::kind::OpKind;

    /// The smallest pool a register file may declare, so pressure tests need
    /// only a handful of values to reach spilling.
    const TEST_FILE: RegisterFile = RegisterFile {
        fixed: &[],
        inputs: [Reg(0), Reg(1), Reg(2), Reg(3)],
        scratch: RegSet::range(4, RegisterFile::MIN_SCRATCH),
        reload: [Reg(11), Reg(12)],
        temps_for: no_temps,
        vector_bytes: 16,
    }
    .checked();

    /// `TEST_FILE`, but every `Neg` asks for a temp — the AVX2 sign-mask case.
    const TEMP_FILE: RegisterFile = RegisterFile {
        temps_for: neg_wants_a_temp,
        ..TEST_FILE
    }
    .checked();

    /// `TEMP_FILE` with headroom above [`RegisterFile::MIN_SCRATCH`].
    ///
    /// The nest tests need a pool that can spare a register to carry, and the
    /// minimum-sized file by construction cannot: the carry budget is
    /// `pool - MIN_SCRATCH`, which is zero there. Ten registers mirrors the
    /// SSE2 tier, whose budget is four.
    const NEST_FILE: RegisterFile = RegisterFile {
        scratch: RegSet::range(4, 7).union(RegSet::of(&[Reg(13), Reg(14), Reg(15)])),
        ..TEMP_FILE
    }
    .checked();

    fn neg_wants_a_temp(op: &ScheduledOp) -> u8 {
        u8::from(matches!(op, ScheduledOp::Unary(OpKind::Neg, _)))
    }

    /// A point in the innermost body — the only scope a loop-free
    /// allocation has.
    fn body(index: usize) -> Point {
        Point {
            scope: Scope::Body,
            index,
        }
    }

    fn def(value: u32, op: ScheduledOp) -> Def {
        Def {
            value: ValueId(value),
            op,
        }
    }

    fn alloc(schedule: Vec<Def>) -> NestAllocation {
        LinearScan.allocate(schedule, &TEST_FILE)
    }

    /// How many of a loop-free allocation's values are in a stack slot.
    fn spill_count(a: &NestAllocation) -> usize {
        a.body()
            .schedule()
            .iter()
            .filter(|d| a.placement(d.value).spills())
            .count()
    }

    /// Where `v` lives at its own definition, in `scope`.
    ///
    /// A placement is a schedule, so every query needs a point; the point a
    /// test means when it asks "where did the allocator put this" is the
    /// definition, and finding it is the schedule's job rather than each
    /// assertion's.
    fn at_def(a: &Allocation<'_>, v: ValueId) -> Where {
        let i = a
            .schedule()
            .iter()
            .position(|d| d.value == v)
            .unwrap_or_else(|| panic!("{v:?} is not in this schedule"));
        a.where_at(v, i)
    }

    /// `at_def` against a loop-free allocation, whose one scope is the body.
    fn at(a: &NestAllocation, v: ValueId) -> Where {
        at_def(&a.body(), v)
    }

    /// `v2 = X + Y`.
    fn add_xy() -> Vec<Def> {
        vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Binary(OpKind::Add, ValueId(0), ValueId(1))),
        ]
    }

    #[test]
    fn an_empty_schedule_allocates_nothing() {
        let a = alloc(vec![]);
        assert!(a.body().schedule().is_empty());
        assert_eq!(spill_count(&a), 0);
    }

    /// The schedule is an output, and for linear scan it is the input order:
    /// the arena's append-only structure already guarantees topological order.
    #[test]
    fn linear_scan_returns_the_order_it_was_given() {
        let a = alloc(add_xy());
        let order: Vec<ValueId> = a.body().schedule().iter().map(|d| d.value).collect();
        assert_eq!(order, vec![ValueId(0), ValueId(1), ValueId(2)]);
    }

    #[test]
    fn vars_are_pinned_to_the_input_registers() {
        let a = alloc(add_xy());
        assert_eq!(at(&a, ValueId(0)), Where::Reg(Reg(0)), "X");
        assert_eq!(at(&a, ValueId(1)), Where::Reg(Reg(1)), "Y");
        // The sum takes the pool, never an input register.
        assert_eq!(at(&a, ValueId(2)), Where::Reg(Reg(4)));
        assert_eq!(spill_count(&a), 0);
    }

    /// Every value the emitter walks has a placement — the invariant that used
    /// to be spread across three parallel maps and checked at runtime.
    #[test]
    fn placement_is_total_over_the_schedule() {
        let a = alloc(add_xy());
        let body = a.body();
        let placed: Vec<Where> = body.schedule().iter().map(|d| at(&a, d.value)).collect();
        assert_eq!(placed.len(), body.schedule().len());
    }

    // -------------------------------------------------------------------------
    // The placement sequence itself
    // -------------------------------------------------------------------------

    /// A one-range placement answers the same thing at every point — which is
    /// what makes today's allocator's output a special case of the sequence
    /// rather than a different kind of answer.
    #[test]
    fn a_single_range_answers_everywhere() {
        let p = Placement::new(Span {
            from: body(0),
            at: Where::Reg(Reg(7)),
        });
        for i in [0usize, 1, 99] {
            assert_eq!(p.at(body(i)), Where::Reg(Reg(7)));
        }
        assert!(!p.spills());
        assert_eq!(p.registers().collect::<Vec<_>>(), vec![Reg(7)]);
    }

    /// The case the whole type exists for: a value in a register up to a
    /// point and in a slot after it. Nothing emits this yet — the policy that
    /// will is the work this API unblocks — so it is checked here directly.
    #[test]
    fn a_ranged_placement_answers_per_point() {
        let p = Placement::new(Span {
            from: body(3),
            at: Where::Reg(Reg(5)),
        })
        .then(Span {
            from: body(9),
            at: Where::Spilled,
        });

        assert_eq!(p.at(body(3)), Where::Reg(Reg(5)), "the first range starts");
        assert_eq!(p.at(body(8)), Where::Reg(Reg(5)), "still the first range");
        assert_eq!(p.at(body(9)), Where::Spilled, "`from` is inclusive");
        assert_eq!(p.at(body(400)), Where::Spilled, "the last range runs on");

        // Total below the definition too: an answer, not a panic.
        assert_eq!(p.at(body(0)), Where::Reg(Reg(5)));
        assert_eq!(p.defined_at(), body(3));

        // A value in a slot for *part* of its life still needs a slot, and the
        // register it held earlier is still a register something wrote.
        assert!(p.spills());
        assert_eq!(p.registers().collect::<Vec<_>>(), vec![Reg(5)]);
    }

    #[test]
    #[should_panic(expected = "names coordinate 4")]
    fn a_var_past_the_input_registers_panics() {
        let _ = alloc(vec![def(0, ScheduledOp::Var(4))]);
    }

    /// Two values whose live ranges do not overlap may share one register —
    /// that is the whole point of tracking last use rather than defs.
    #[test]
    fn disjoint_live_ranges_share_a_register() {
        let a = alloc(vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Binary(OpKind::Add, ValueId(0), ValueId(1))),
            def(3, ScheduledOp::Unary(OpKind::Neg, ValueId(2))),
            def(4, ScheduledOp::Unary(OpKind::Abs, ValueId(3))),
        ]);
        assert_eq!(spill_count(&a), 0);
        // v2's last use is v3, so v4 may reuse v2's register.
        assert_eq!(at(&a, ValueId(2)), at(&a, ValueId(4)));
    }

    // -------------------------------------------------------------------------
    // Instruction temps
    // -------------------------------------------------------------------------

    /// The temp is a real register from the pool, and it is nobody's operand
    /// and not the destination — the encoder writes it while the operands are
    /// still live and before it writes `dst`.
    #[test]
    fn a_temp_collides_with_neither_the_operand_nor_the_destination() {
        let schedule = vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Unary(OpKind::Neg, ValueId(0))),
        ];
        let a = LinearScan.allocate(schedule, &TEMP_FILE);

        let temp = a.body().scratch(1).temp(0).expect("`Neg` asked for a temp");
        assert!(
            TEMP_FILE.scratch.contains(temp),
            "{temp:?} is not the pool's"
        );
        assert_ne!(Where::Reg(temp), at(&a, ValueId(0)));
        assert_ne!(Where::Reg(temp), at(&a, ValueId(1)));
    }

    /// A `Select` gets its own third reload target, disjoint from its operands,
    /// its destination and its encoding temp — the four registers the
    /// instruction can be using at once.
    #[test]
    fn a_select_reserves_a_third_reload_target() {
        let schedule = vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Binary(OpKind::Lt, ValueId(0), ValueId(1))),
            def(3, ScheduledOp::Binary(OpKind::Add, ValueId(0), ValueId(1))),
            def(4, ScheduledOp::Binary(OpKind::Sub, ValueId(0), ValueId(1))),
            def(
                5,
                ScheduledOp::Ternary(OpKind::Select, ValueId(2), ValueId(3), ValueId(4)),
            ),
        ];
        let a = LinearScan.allocate(schedule, &TEMP_FILE);
        let s = a.body().scratch(5);

        let arm = s.arm_reload.expect("a `Select` always reserves one");
        assert!(TEMP_FILE.scratch.contains(arm), "{arm:?} is not the pool's");
        for v in [ValueId(2), ValueId(3), ValueId(4), ValueId(5)] {
            assert_ne!(
                Where::Reg(arm),
                at(&a, v),
                "{arm:?} is still holding {v:?} when the Select reloads into it"
            );
        }
        assert_ne!(Some(arm), s.temp(0), "the two roles must be two registers");
    }

    /// Only a `Select` asks: every other shape leaves the register alone.
    #[test]
    fn nothing_but_a_select_reserves_an_arm_reload() {
        let a = LinearScan.allocate(add_xy(), &TEMP_FILE);
        assert_eq!(a.body().scratch(2).arm_reload, None);
    }

    /// A pool shrunk below what an encoding needs is not a smaller budget, it
    /// is an instruction with nowhere to put its temp — so `capped` holds the
    /// floor rather than letting `max_regs` reach through it.
    #[test]
    fn capping_the_pool_stops_at_the_floor() {
        let tiny = TEMP_FILE.capped(Some(1));
        assert_eq!(tiny.scratch.len(), RegisterFile::MIN_SCRATCH);

        // The shape that used to fall through: a `Neg` whose operand is a
        // computed value, so the operand is pool-resident rather than
        // pre-colored into an input register.
        let a = LinearScan.allocate(
            vec![
                def(0, ScheduledOp::Var(0)),
                def(1, ScheduledOp::Var(1)),
                def(2, ScheduledOp::Binary(OpKind::Add, ValueId(0), ValueId(1))),
                def(3, ScheduledOp::Unary(OpKind::Neg, ValueId(2))),
            ],
            &tiny,
        );
        let temp = a.body().scratch(3).temp(0).expect("`Neg` asked for a temp");
        assert_ne!(Where::Reg(temp), at(&a, ValueId(2)));
    }

    /// Only the ops that asked get one — the register is the allocator's
    /// everywhere else, which is the whole reason for asking per-op.
    #[test]
    fn an_op_that_asks_for_no_temp_gets_none() {
        let a = LinearScan.allocate(add_xy(), &TEMP_FILE);
        assert_eq!(a.body().scratch(2).temp(0), None, "`Add` asked for no temp");
    }

    /// The temp's live range is one instruction: the next one may take the
    /// same register for its result.
    #[test]
    fn a_temp_is_free_again_at_the_next_instruction() {
        let schedule = vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Unary(OpKind::Neg, ValueId(0))),
            def(2, ScheduledOp::Unary(OpKind::Sqrt, ValueId(1))),
            def(3, ScheduledOp::Unary(OpKind::Sqrt, ValueId(2))),
            def(4, ScheduledOp::Unary(OpKind::Sqrt, ValueId(3))),
        ];
        let a = LinearScan.allocate(schedule, &TEMP_FILE);
        assert_eq!(spill_count(&a), 0, "four values fit a four-register pool");

        let temp = a.body().scratch(1).temp(0).expect("`Neg` asked for a temp");
        let reused = a
            .body()
            .schedule()
            .iter()
            .any(|d| at(&a, d.value) == Where::Reg(temp));
        assert!(reused, "{temp:?} went back to the pool after the `Neg`");
    }

    /// Six values live at once cannot fit a four-register pool.
    #[test]
    fn pressure_beyond_the_pool_spills() {
        let mut schedule = vec![def(0, ScheduledOp::Var(0))];
        for i in 1..=6u32 {
            schedule.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        let mut acc = ValueId(1);
        for i in 2..=6u32 {
            schedule.push(def(
                6 + i,
                ScheduledOp::Binary(OpKind::Add, acc, ValueId(i)),
            ));
            acc = ValueId(6 + i);
        }
        let a = alloc(schedule);
        assert!(spill_count(&a) > 0);
        // A placement is one choice, so nothing is both in a register and on
        // the stack — the miscompile the three parallel maps could express.
        for d in a.body().schedule() {
            if a.placement(d.value).spills() {
                assert_eq!(at(&a, d.value), Where::Spilled);
            }
        }
    }

    /// A constant under pressure is rematerialized, never spilled: re-emitting
    /// the load beats a store plus a reload.
    #[test]
    fn constants_are_rematerialized_rather_than_spilled() {
        let mut schedule = vec![def(0, ScheduledOp::Var(0))];
        for i in 1..=6u32 {
            schedule.push(def(i, ScheduledOp::Const(i as f32)));
        }
        let mut acc = ValueId(1);
        for i in 2..=6u32 {
            schedule.push(def(
                6 + i,
                ScheduledOp::Binary(OpKind::Add, acc, ValueId(i)),
            ));
            acc = ValueId(6 + i);
        }
        let a = alloc(schedule);

        let remat: Vec<(ValueId, u32)> = (1..=6u32)
            .filter_map(|i| match at(&a, ValueId(i)) {
                Where::Remat(bits) => Some((ValueId(i), bits)),
                _ => None,
            })
            .collect();
        assert!(!remat.is_empty(), "constants under pressure should remat");
        assert_eq!(spill_count(&a), 0, "no constant belongs in a spill slot");
        for (vid, bits) in remat {
            assert_eq!(
                bits,
                (vid.0 as f32).to_bits(),
                "{vid:?} rematerializes the wrong constant"
            );
        }
    }

    /// Belady: with no constants in play, the value used farthest in the
    /// future is the one that goes to memory.
    ///
    /// The scenario separates Belady from FIFO and LRU deliberately. Four
    /// values fill the pool in the order v1..v4, then v5 forces an eviction —
    /// but they are *consumed* in that same order, so v1 is simultaneously the
    /// oldest, the least recently used, and the one needed soonest. FIFO and
    /// LRU both evict v1. Only a rule that looks forward evicts v4.
    #[test]
    fn belady_evicts_the_value_used_farthest_out() {
        // One more independent value than the pool holds, so exactly one must
        // go to memory and the test is about *which*.
        let live = u32::from(RegisterFile::MIN_SCRATCH) + 1;
        let mut schedule = vec![def(0, ScheduledOp::Var(0))];
        for i in 1..=live {
            schedule.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        // Consume v1 first, then v2, v3, … and v`live-1` last.
        let mut acc = ValueId(live);
        for i in 1..live {
            schedule.push(def(
                100 + i,
                ScheduledOp::Binary(OpKind::Add, acc, ValueId(i)),
            ));
            acc = ValueId(100 + i);
        }
        let a = alloc(schedule);

        assert_eq!(
            at(&a, ValueId(live - 1)),
            Where::Spilled,
            "v{live_minus_1} is needed last, so it is the one to evict",
            live_minus_1 = live - 1
        );
        assert!(
            matches!(at(&a, ValueId(1)), Where::Reg(_)),
            "v1 is needed next, so it must keep its register — evicting it is \
             what FIFO and LRU would have done"
        );
    }

    /// Purity is load-bearing: the collapse driver sizes a frame with one run
    /// and emits into it with another, and a disagreement misplaces every slot.
    #[test]
    fn allocation_is_deterministic() {
        let a = alloc(add_xy());
        let b = alloc(add_xy());
        for d in a.body().schedule() {
            assert_eq!(at(&a, d.value), at(&b, d.value));
        }
        assert_eq!(spill_count(&a), spill_count(&b));
    }

    /// A hoisted value is pinned to the slot its prologue parked it in,
    /// overriding whatever the allocator gave the placeholder def.
    #[test]
    fn a_placement_can_be_overridden() {
        let mut a = alloc(add_xy());
        assert!(matches!(at(&a, ValueId(2)), Where::Reg(_)));
        a.place(ValueId(2), Where::Spilled);
        assert_eq!(at(&a, ValueId(2)), Where::Spilled);
        assert_eq!(spill_count(&a), 1);
    }

    /// All three `Ternary` operands count toward liveness. Missing one frees a
    /// register that is still in use.
    #[test]
    fn every_ternary_operand_extends_liveness() {
        let sel = ScheduledOp::Ternary(OpKind::Select, ValueId(0), ValueId(1), ValueId(2));
        assert_eq!(
            operands(&sel).collect::<Vec<_>>(),
            vec![ValueId(0), ValueId(1), ValueId(2)]
        );
        let a = alloc(vec![
            def(0, ScheduledOp::Var(0)),
            def(1, ScheduledOp::Var(1)),
            def(2, ScheduledOp::Var(2)),
            def(3, sel),
        ]);
        assert_eq!(spill_count(&a), 0);
    }

    /// A destination never lands in a register one of its own operands is
    /// still living in — the invariant `resolve_operands` reads back off the
    /// allocation, and the reason SSE2 can write its two-operand form
    /// directly.
    ///
    /// `dst op= right` corrupts `right` when `dst == right` and `dst != left`.
    /// That assignment is unrepresentable here: a destination takes a slot
    /// with no live owner, or evicts one — and an evicted value's placement
    /// becomes `Spilled` for its whole life, so it is read back from memory
    /// rather than from the register the destination took. The backend needs
    /// no stashing temp to route around a case that cannot arise, which is
    /// what `emit_binary_safe` used to be and what held xmm10 out of every
    /// kernel's pool.
    #[test]
    fn a_destination_never_lands_on_a_resident_operand() {
        // Wide enough to evict: `width` values all live at once over a pool of
        // `MIN_SCRATCH`, then folded pairwise so every fold reads two of them.
        let width = u32::from(RegisterFile::MIN_SCRATCH) * 3;
        let mut schedule = vec![def(0, ScheduledOp::Var(0))];
        for i in 1..=width {
            schedule.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        let mut acc = ValueId(1);
        for i in 2..=width {
            schedule.push(def(
                width + i,
                ScheduledOp::Binary(OpKind::Sub, acc, ValueId(i)),
            ));
            acc = ValueId(width + i);
        }

        // Both files: `TEMP_FILE` also reserves a temp before placing the
        // destination, which is the other way a destination could be pushed
        // onto an operand.
        for file in [&TEST_FILE, &TEMP_FILE] {
            let a = LinearScan.allocate(schedule.clone(), file);
            assert!(
                spill_count(&a) > 0,
                "the schedule has to reach eviction for this to test anything"
            );
            // Pairs where both ends are the pool's — the case the two-operand
            // form would corrupt. An input register can never collide with a
            // destination (the pool excludes them), so counting those would
            // let the test pass vacuously.
            let mut contested = 0;
            let body = a.body();
            for (i, d) in body.schedule().iter().enumerate() {
                // At the instruction's own point: that is where a destination
                // and its operands would collide.
                let Where::Reg(dst) = body.where_at(d.value, i) else {
                    continue; // Spilled: the destination is `reload[0]`, not the pool's.
                };
                for operand in operands(&d.op) {
                    let at = body.where_at(operand, i);
                    assert_ne!(
                        at,
                        Where::Reg(dst),
                        "{:?}: destination {dst:?} is where its operand {operand:?} lives",
                        d.value
                    );
                    if matches!(at, Where::Reg(r) if file.scratch.contains(r)) {
                        contested += 1;
                    }
                }
            }
            assert!(
                contested > 0,
                "no instruction read a pool-resident operand, so nothing above \
                 could have collided"
            );
        }
    }

    /// A carried register is untouched by every scope inside the loop.
    ///
    /// This is the whole safety property of showing the allocator the nest. A
    /// value the outer region leaves in a register is read by the body on
    /// every iteration, so anything the body writes there is a miscompile that
    /// only shows up as wrong pixels. The body's pool excludes carries by
    /// construction (`RegisterFile::inside`) — this is what says so out loud,
    /// and it checks the *temps* too, which are pool registers no `Placement`
    /// records.
    #[test]
    fn a_carried_register_is_untouched_by_everything_inside_the_loop() {
        // An outer region computing invariants, and a body that reads them.
        let mut outer = vec![def(0, ScheduledOp::Var(1))];
        let width = 6u32;
        for i in 1..=width {
            outer.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        let roots: Vec<ValueId> = (1..=width).map(ValueId).collect();

        let mut body = vec![def(100, ScheduledOp::Var(0))];
        let mut acc = ValueId(100);
        for (i, root) in roots.iter().enumerate() {
            body.push(def(
                200 + i as u32,
                ScheduledOp::Binary(OpKind::Add, acc, *root),
            ));
            acc = ValueId(200 + i as u32);
        }

        let nest = ScopedSchedule {
            regions: vec![ScopeRegion {
                roots: roots.clone(),
                schedule: outer,
            }],
            body,
        };
        let alloc = LinearScan.allocate_nest(nest, &NEST_FILE);

        let carries: Vec<(ValueId, Reg)> = roots
            .iter()
            .filter_map(|v| alloc.carried(*v).map(|r| (*v, r)))
            .collect();
        assert!(
            !carries.is_empty(),
            "nothing was carried, so this test asserts nothing about carrying"
        );

        // Every register any scope inside the loop can write. A carried root
        // is *not* one of them: its placement inside the loop is the carry,
        // and the register it held in the producing region belongs to that
        // region — so the roots are excluded rather than counted.
        let body = alloc.body();
        let mut inside: Vec<Reg> = body
            .schedule()
            .iter()
            .filter(|d| !roots.contains(&d.value))
            .flat_map(|d| alloc.placement(d.value).registers())
            .collect();
        for i in 0..body.schedule().len() {
            let s = body.scratch(i);
            inside.extend((0..Scratch::MAX_TEMPS).filter_map(|k| s.temp(k)));
            inside.extend(s.arm_reload);
        }

        for (vid, carry) in &carries {
            assert!(
                !inside.contains(carry),
                "{vid:?} is carried in {carry:?}, which the body also writes"
            );
        }
    }

    /// A root's placement is the whole of what used to need a `carries` map
    /// beside it: its register inside the region that computes it, and then —
    /// from the first point of the loops within — either the carry or a slot.
    #[test]
    fn a_root_is_placed_twice_and_says_for_itself_whether_it_is_carried() {
        let mut outer = vec![def(0, ScheduledOp::Var(1))];
        let width = 6u32;
        for i in 1..=width {
            outer.push(def(i, ScheduledOp::Unary(OpKind::Neg, ValueId(0))));
        }
        let roots: Vec<ValueId> = (1..=width).map(ValueId).collect();

        let mut body = vec![def(100, ScheduledOp::Var(0))];
        let mut acc = ValueId(100);
        for (i, root) in roots.iter().enumerate() {
            body.push(def(
                200 + i as u32,
                ScheduledOp::Binary(OpKind::Add, acc, *root),
            ));
            acc = ValueId(200 + i as u32);
        }

        let alloc = LinearScan.allocate_nest(
            ScopedSchedule {
                regions: vec![ScopeRegion {
                    roots: roots.clone(),
                    schedule: outer,
                }],
                body,
            },
            &NEST_FILE,
        );

        let (mut carried, mut parked) = (0, 0);
        for root in &roots {
            let p = alloc.placement(*root);
            assert_eq!(
                p.defined_at().scope,
                Scope::Region(0),
                "{root:?} is computed by the outer region"
            );
            // Inside the region: whatever the scan chose, and a pool register
            // there — never the carry, which is picked from what the region
            // leaves free.
            let inside_region = p.at(Point {
                scope: Scope::Region(0),
                index: usize::MAX,
            });
            let in_the_loop = p.at(Point {
                scope: Scope::Body,
                index: 0,
            });
            assert_ne!(
                inside_region, in_the_loop,
                "{root:?} would need only one range, but a root always changes \
                 place at the loop it is read inside"
            );
            match alloc.carried(*root) {
                Some(reg) => {
                    assert_eq!(in_the_loop, Where::Reg(reg));
                    carried += 1;
                }
                None => {
                    assert_eq!(in_the_loop, Where::Spilled);
                    assert!(p.spills(), "a parked root is in a slot");
                    parked += 1;
                }
            }
        }
        assert!(carried > 0, "the file has budget, so something is carried");
        assert!(
            parked > 0,
            "the budget is smaller than the root count, so something is parked"
        );
    }

    /// `checked()` is the isolation's teeth: a file whose reload register sits
    /// inside the allocatable pool would let a reload clobber a live value.
    #[test]
    #[should_panic(expected = "reload register is inside the allocatable pool")]
    fn a_reload_register_inside_the_pool_is_refused() {
        let _refused = RegisterFile {
            scratch: RegSet::range(4, 8), // swallows Reg(11)
            ..TEST_FILE
        }
        .checked();
    }

    /// The same teeth for a backend's own scratch: `fixed` is declared so this
    /// is a const-eval failure rather than an argument in a comment.
    #[test]
    #[should_panic(expected = "fixed backend scratch register is inside the allocatable pool")]
    fn a_fixed_register_inside_the_pool_is_refused() {
        let _refused = RegisterFile {
            fixed: &[Reg(5)], // inside TEST_FILE's 4..8 pool
            ..TEST_FILE
        }
        .checked();
    }

    /// A set is not a range: the pool may hold registers on both sides of a
    /// reserved one, which is the whole reason `RegSet` replaced base+count.
    #[test]
    fn the_pool_may_straddle_a_reserved_register() {
        let straddling = RegisterFile {
            scratch: RegSet::of(&[Reg(4), Reg(5), Reg(6), Reg(7), Reg(14), Reg(15)]),
            ..TEST_FILE
        }
        .checked();
        assert_eq!(straddling.scratch.len(), 6);
        let regs: alloc::vec::Vec<Reg> = straddling.scratch.iter().collect();
        assert_eq!(
            regs,
            alloc::vec![Reg(4), Reg(5), Reg(6), Reg(7), Reg(14), Reg(15)]
        );
    }

    /// `capped` shrinks a non-contiguous pool to its lowest members.
    #[test]
    fn capping_takes_the_lowest_members() {
        let set = RegSet::of(&[Reg(4), Reg(9), Reg(14), Reg(15)]);
        let regs: alloc::vec::Vec<Reg> = set.take(2).iter().collect();
        assert_eq!(regs, alloc::vec![Reg(4), Reg(9)]);
        assert_eq!(set.take(99).len(), 4, "capping never grows the pool");
    }

    #[test]
    #[should_panic(expected = "input register is inside the allocatable pool")]
    fn an_input_register_inside_the_pool_is_refused() {
        let _refused = RegisterFile {
            inputs: [Reg(0), Reg(1), Reg(2), Reg(4)],
            ..TEST_FILE
        }
        .checked();
    }

    #[test]
    #[should_panic(expected = "vector_bytes")]
    fn a_non_power_of_two_vector_width_is_refused() {
        let _refused = RegisterFile {
            vector_bytes: 24,
            ..TEST_FILE
        }
        .checked();
    }
}
