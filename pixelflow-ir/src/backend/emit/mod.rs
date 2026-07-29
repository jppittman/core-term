//! JIT code emission for expression trees and DAGs.
//!
//! Two register allocation strategies:
//!
//! ## Sethi-Ullman (for trees)
//!
//! Register allocation emerges from tree structure.
//! Sethi-Ullman labeling computes minimum registers needed.
//! Register assignment is a FUNCTION of tree position (depth), not stateful allocation.
//!
//! ```text
//! emit : Expr × Depth → (Code, Reg)
//! ```
//!
//! No explicit alloc/free - the recursion depth IS the register.
//!
//! ## Linear Scan (for DAGs)
//!
//! For expressions with shared subexpressions (from e-graph extraction),
//! linear scan with Belady eviction handles register allocation in O(n×k).
//! With 22 scratch registers and typical live sets of 10-15, this is
//! effectively O(n) and produces near-optimal allocations.
//! See [`regalloc`] module.
//!
//! ## Spilling
//!
//! When `max_regs` is set lower than needed, we spill to stack:
//! - Spilled values stored via STR to [SP, #offset]
//! - Reloaded via LDR to dedicated reload register before use
//! - This lets ML models learn register pressure vs spill tradeoffs

pub mod aarch64;
pub mod avx2;
pub mod avx512;
#[cfg(test)]
pub(crate) mod coverage;
pub mod executable;
pub mod lowering;
pub mod regalloc;
pub mod x86_64;

use crate::kind::OpKind;

#[cfg(target_arch = "x86_64")]
use crate::arena::{ExprArena, ExprId};

use alloc::vec::Vec;

/// Constant pool: maps f32 bit patterns to pool indices.
///
/// Non-zero, non-FMOV-encodable constants are stored in a data section after
/// the RET instruction. Each entry is 16 bytes (the f32 splatted 4x to fill
/// a 128-bit NEON register). During code emission, these constants are loaded
/// with a single `LDR Qt, [X17, #offset]` instead of the 3-instruction
/// MOVZ+MOVK+DUP sequence.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) struct ConstPool {
    /// Deduplicated entries: f32 bit patterns in pool order.
    entries: Vec<u32>,
    /// Map from f32 bits → pool index.
    index: alloc::collections::BTreeMap<u32, u16>,
}

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
impl ConstPool {
    /// Create an empty constant pool.
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
            index: alloc::collections::BTreeMap::new(),
        }
    }

    /// Insert an f32 into the pool (deduplicating by bit pattern) and return
    /// the byte offset for an `LDR Qt, [X17, #offset]` load.
    ///
    /// Zero and FMOV-encodable constants are NOT filtered here — callers
    /// that want the fast path should check `needs_const_pool` first.
    /// Builtin emitters call this unconditionally because every constant
    /// they use benefits from the pool (they are transcendental coefficients,
    /// never zero or FMOV-encodable).
    pub(crate) fn push_f32(&mut self, val: f32) -> Result<u16, &'static str> {
        let bits = val.to_bits();
        if let Some(&idx) = self.index.get(&bits) {
            return Ok(idx * 16);
        }
        let idx = self.entries.len();
        if idx >= 4096 {
            return Err(
                "constant pool overflow: exceeded 12-bit LDR offset limit (max 4095 entries)",
            );
        }
        self.entries.push(bits);
        self.index.insert(bits, idx as u16);
        Ok((idx * 16) as u16)
    }

    /// Get the byte offset for a constant, or None if it's not in the pool.
    fn offset_for(&self, val_bits: u32) -> Option<u16> {
        self.index.get(&val_bits).map(|&idx| idx * 16)
    }

    /// Returns true if the pool has any entries.
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Emit a constant load, using the constant pool when available.
///
/// Falls back to `emit_fmov_imm` for zero and FMOV-encodable values.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn emit_const_load(code: &mut Vec<u8>, dst: Reg, val_bits: u32, pool: &ConstPool) {
    if let Some(offset) = pool.offset_for(val_bits) {
        aarch64::emit_ldr_q_x17(code, dst, offset);
    } else {
        let scratch = [Reg(28), Reg(29), Reg(30), Reg(31)];
        aarch64::emit_fmov_imm(code, dst, f32::from_bits(val_bits), scratch);
    }
}

/// Physical register index.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Reg(pub u8);

/// Location of a value: either in a register or spilled to stack.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Loc {
    /// Value is in a register.
    Reg(Reg),
    /// Value is spilled to stack at this offset from SP.
    Spill(u32),
}

impl Loc {
    /// Get the register, panicking if spilled.
    #[must_use]
    pub fn reg(self) -> Reg {
        match self {
            Loc::Reg(r) => r,
            Loc::Spill(off) => panic!("expected register, got spill slot {}", off),
        }
    }
}

/// Stack frame layout computed from register allocation.
///
/// Pure data: computed from the list of spilled values.
/// Maps each spilled ValueId to its stack offset and computes
/// the total frame size (16-byte aligned for AArch64 SP).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameLayout {
    /// Map from spilled ValueId to stack offset.
    pub spill_slots: alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    /// Total frame size (16-byte aligned).
    pub frame_size: u32,
}

impl FrameLayout {
    /// Compute frame layout from spilled values.
    /// Pure function: spilled list → layout.
    pub fn from_allocation(spilled: &[regalloc::ValueId]) -> Result<Self, &'static str> {
        use alloc::collections::BTreeMap;
        // 2MB max frame — generous but prevents runaway allocations.
        const MAX_FRAME: u32 = 2 * 1024 * 1024;
        let mut spill_slots = BTreeMap::new();
        let mut offset = 0u32;
        for &v in spilled {
            if offset > MAX_FRAME - 16 {
                return Err("spill frame overflow: exceeds 2MB stack limit");
            }
            spill_slots.insert(v, offset);
            offset += 16; // 128-bit vector
        }
        let aligned = (offset + 15) & !15;
        Ok(Self {
            spill_slots,
            frame_size: aligned,
        })
    }
}

/// A concrete instruction to emit, with all registers resolved.
/// Pure data — no side effects, no mutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedOp {
    /// No-op (variable already in input register).
    Nop,
    /// Load constant into dst.
    LoadConst { dst: Reg, val_bits: u32 },
    /// Unary: dst = op(src).
    Unary { op: OpKind, dst: Reg, src: Reg },
    /// Integer shift by a compile-time immediate: dst = src `op` amount, where
    /// `op` is `Shl` or `Shr` (the hardware shift encoders are imm-only).
    ShiftImm {
        op: OpKind,
        dst: Reg,
        src: Reg,
        amount: u8,
    },
    /// Binary: dst = op(left, right).
    Binary {
        op: OpKind,
        dst: Reg,
        left: Reg,
        right: Reg,
    },
    /// Fused multiply-add via FMLA: dst = c + a*b.
    /// Requires dst to hold c before FMLA.
    FusedMulAdd { dst: Reg, a: Reg, b: Reg },
    /// Decomposed multiply-add: FMUL(dst, a, b) then reload c, then FADD(dst, dst, c).
    /// Used when a and b are both spilled (can't load both + c simultaneously).
    /// `c_deferred`: if Some, c must be reloaded *after* FMUL.
    DecomposedMulAdd {
        dst: Reg,
        a: Reg,
        b: Reg,
        c: Reg,
        c_deferred: Option<DeferredReload>,
    },
    /// BSL select: dst = mask ? if_true : if_false (mask pre-loaded into dst).
    Select {
        dst: Reg,
        if_true: Reg,
        if_false: Reg,
    },
    /// Bound-memory gather: `dst = buffer[slot][idx_lane]`. Emitted only on the
    /// AVX-512 backend (native `vgatherdps`); other backends reject it. The
    /// buffer base pointer is loaded from the context struct (rdi) at
    /// `slot * 8`.
    Gather { dst: Reg, idx: Reg, slot: u16 },
}

/// A deferred reload: value loaded mid-instruction (after a partial computation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeferredReload {
    /// Load from stack at the given SP offset.
    FromStack(u32),
    /// Rematerialize a constant.
    Const(u32),
}

/// Reload instruction: load a value into a register.
///
/// Either reload from stack (spilled) or rematerialize a constant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reload {
    /// Load from stack at the given SP offset.
    FromStack { target: Reg, offset: u32 },
    /// Rematerialize a constant (emit FMOV immediate).
    Const { target: Reg, val_bits: u32 },
}

/// Store instruction: spill computed value to stack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Store {
    pub src: Reg,
    pub offset: u32,
}

/// Fully resolved instruction with reloads and optional store.
#[derive(Clone, Debug)]
pub struct InstructionPlan {
    /// Reloads to emit before the main op.
    pub reloads: Vec<Reload>,
    /// The main operation.
    pub op: ResolvedOp,
    /// Optional MOV to set up accumulator/mask before main op.
    pub setup_mov: Option<(Reg, Reg)>,
    /// Store to emit after the main op (if dst is spilled).
    pub store: Option<Store>,
}

/// Emission context with register budget for ML training.
#[derive(Clone, Debug)]
pub struct EmitCtx {
    /// Maximum scratch registers before spilling (ML parameter).
    /// Default: 10 (v16-v25 on ARM64; v8-v15 are callee-saved and excluded —
    /// see `SCRATCH_BASE`).
    pub max_regs: u8,
    /// Current spill offset from SP.
    pub spill_offset: u32,
    /// Number of spills performed (for cost modeling).
    pub spill_count: u32,
}

impl Default for EmitCtx {
    fn default() -> Self {
        Self {
            max_regs: 10, // v16-v25 on ARM64 (v8-v15 callee-saved & excluded, v26-v27 reserved for RELOAD_REGS)
            spill_offset: 0,
            spill_count: 0,
        }
    }
}

impl EmitCtx {
    /// Create context with custom register budget.
    #[must_use]
    pub fn with_max_regs(max_regs: u8) -> Self {
        Self {
            max_regs,
            ..Default::default()
        }
    }

    /// Allocate a spill slot, returns offset.
    pub fn alloc_spill(&mut self) -> u32 {
        let off = self.spill_offset;
        self.spill_offset += 16; // 128-bit vector
        self.spill_count += 1;
        off
    }
}

/// Input registers: X=v0, Y=v1, Z=v2, W=v3
pub const INPUT_REGS: [Reg; 4] = [Reg(0), Reg(1), Reg(2), Reg(3)];

/// First scratch register (after inputs), x86-64.
///
/// SysV has no callee-saved XMM registers, so any register past the inputs
/// is fair game.
#[cfg(target_arch = "x86_64")]
pub const SCRATCH_BASE: u8 = 4;

/// First scratch register (after inputs), aarch64.
///
/// AAPCS64 callee-saves the low 64 bits of v8-v15. The kernels here are leaf
/// functions emitted with no prologue/epilogue that preserves them, so the
/// generic scratch pool must steer clear of that range entirely — handing one
/// of v8-v15 to `linear_scan` as a scratch register would silently corrupt
/// whatever the *caller* had live there across the JIT call.
#[cfg(target_arch = "aarch64")]
pub const SCRATCH_BASE: u8 = 16;

/// Reload registers for spilled values.
///
/// When an operation's operands are spilled to stack, each is loaded into a
/// reload register. Two registers suffice: the destination register doubles
/// as a temporary for one operand (ARM reads all sources before writing dest).
///
/// Layout (ARM64):
///   v0-v3:   INPUT_REGS (X, Y, Z, W)
///   v8-v15:  callee-saved; never handed out by the generic scratch pool
///            (see `SCRATCH_BASE`)
///   v16-v25: allocatable scratch (max_regs=10)
///   v26-v27: RELOAD_REGS (spill reload area)
///   v28-v31: fixed-purpose scratch (select-guard reduction, imm construction)
#[cfg(target_arch = "aarch64")]
pub const RELOAD_REGS: [Reg; 2] = [Reg(26), Reg(27)];

/// For backward compatibility with the tree-walk emitter.
#[cfg(target_arch = "aarch64")]
pub const RELOAD_REG: Reg = Reg(27);

/// Dedicated register for reloading spilled values (x86-64: xmm12)
#[cfg(target_arch = "x86_64")]
pub const RELOAD_REG: Reg = Reg(12);

/// x86-64 reload registers (placeholder — x86 DAG path not yet implemented).
#[cfg(target_arch = "x86_64")]
pub const RELOAD_REGS: [Reg; 2] = [Reg(11), Reg(12)];

// =============================================================================
// Functional Emitter (x86-64)
// =============================================================================

// =============================================================================
// High-level API
// =============================================================================

/// Compile result with metadata for ML training.
pub struct CompileResult {
    /// The executable code.
    pub code: executable::ExecutableCode,
    /// Number of spills performed.
    pub spill_count: u32,
    /// Total stack space used for spills (bytes).
    pub spill_bytes: u32,
    /// Register budget that was used.
    pub max_regs: u8,
    /// X-invariant values hoisted out of the collapse loop into the
    /// once-per-call prologue (0 for per-batch kernels, and for collapse
    /// kernels with nothing to hoist).
    pub hoisted_values: u32,
}

/// Compile an [`ExprArena`] DAG to executable code.
///
/// This is the arena counterpart of [`compile`]. The arena's topological node
/// order eliminates the linearization pass entirely.
#[cfg(target_arch = "aarch64")]
pub fn compile_arena(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
) -> Result<executable::ExecutableCode, &'static str> {
    compile_arena_dag(arena, root).map(|r| r.code)
}

/// Compile an [`ExprArena`] DAG using graph coloring / linear scan register allocation.
///
/// The arena IS the linearized schedule, so the linearization pass is free:
/// `ExprId` maps 1:1 to `ValueId` for reachable nodes.
#[cfg(target_arch = "aarch64")]
pub fn compile_arena_dag(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
) -> Result<CompileResult, &'static str> {
    compile_arena_dag_with_ctx(arena, root, EmitCtx::default())
}

/// Compile an [`ExprArena`] DAG with explicit register budget.
///
/// The arena's append-only structure guarantees topological order, so the
/// schedule comes for free -- `ExprId` maps 1:1 to `ValueId` for reachable nodes.
#[cfg(target_arch = "aarch64")]
pub fn compile_arena_dag_with_ctx(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
    ctx: EmitCtx,
) -> Result<CompileResult, &'static str> {
    // Lower derivatives, reductions, memory reads, then transcendentals to
    // primitives (one source of truth in `lowering`); the aarch64 emitter then
    // never sees Dwrt/Reduce/Gather/Sin/Cos/etc. Dwrt runs first: its chain
    // rule emits Sin/Cos/Exp nodes that the transcendental pass must still
    // lower. (aarch64 gather emission is a later slice.)
    let (arena, root) = lowering::lower_dwrt_owned(arena, root)?;
    let (arena, root) = lowering::expand_reduce_owned(&arena, root);
    let (arena, root) = lowering::expand_gather_owned(&arena, root);
    let (arena, root) = lowering::expand_transcendentals_owned(&arena, root);
    let arena = &arena;
    let schedule = arena_to_schedule(arena, root);
    let uses_map = arena_to_uses(&schedule);
    compile_from_schedule(schedule, uses_map, ctx)
}

// =============================================================================
// Amortized compilation workspace
// =============================================================================

/// Pre-allocated workspace for amortized JIT compilation.
///
/// Owns a reusable [`executable::CodeBuffer`] and pre-sized scratch vectors,
/// eliminating mmap/munmap syscalls and Vec allocations on every compile.
///
/// # Usage
///
/// ```ignore
/// let mut ws = CompileWorkspace::new(65536)?;
/// loop {
///     let func = ws.compile_arena(&arena, root)?;
///     // Use func... (valid until next compile_arena call)
/// }
/// ```
///
/// # Safety
///
/// The returned `KernelFn` is invalidated by the next call to `compile_arena`.
/// The caller must not hold references across compiles.
#[cfg(target_arch = "aarch64")]
pub struct CompileWorkspace {
    /// Reusable executable memory region.
    code_buf: executable::CodeBuffer,
    /// Scratch buffer for schedule entries.
    schedule: Vec<(regalloc::ValueId, ScheduledOp)>,
    /// Scratch buffer for uses map.
    uses_map: Vec<Vec<regalloc::ValueId>>,
    /// Scratch buffer for reachability marking.
    reachable: Vec<bool>,
    /// Scratch buffer for ExprId -> ValueId mapping.
    id_map: Vec<regalloc::ValueId>,
    /// Register budget.
    ctx: EmitCtx,
}

#[cfg(target_arch = "aarch64")]
impl CompileWorkspace {
    /// Create a new workspace with the given code buffer capacity (bytes).
    ///
    /// `code_capacity` is rounded up to the system page size. 64KB is
    /// generous for expressions up to ~500 nodes.
    pub fn new(code_capacity: usize) -> Result<Self, &'static str> {
        Ok(Self {
            code_buf: executable::CodeBuffer::new(code_capacity)?,
            schedule: Vec::with_capacity(256),
            uses_map: Vec::with_capacity(256),
            reachable: Vec::with_capacity(256),
            id_map: Vec::with_capacity(256),
            ctx: EmitCtx::default(),
        })
    }

    /// Set the register budget for subsequent compiles.
    pub fn set_max_regs(&mut self, max_regs: u8) {
        self.ctx.max_regs = max_regs;
    }

    /// Compile an arena DAG, returning a function pointer into reusable memory.
    ///
    /// # Safety
    ///
    /// The returned function pointer is valid only until the next call to
    /// `compile_arena`. The caller must ensure no concurrent use.
    pub unsafe fn compile_arena(
        &mut self,
        arena: &crate::arena::ExprArena,
        root: crate::arena::ExprId,
    ) -> Result<executable::KernelFn, &'static str> {
        // 1. Build schedule into pre-allocated buffers.
        self.schedule.clear();
        let len = arena.len();

        // Grow scratch buffers if needed (retains capacity across compiles).
        self.reachable.clear();
        self.reachable.resize(len, false);
        self.id_map.clear();
        self.id_map.resize(len, regalloc::ValueId(u32::MAX));

        mark_reachable(arena, root, &mut self.reachable);

        let id_map = &mut self.id_map;
        let schedule = &mut self.schedule;
        let reachable = &self.reachable;

        let mut next_id = 0u32;
        for idx in 0..len {
            if !reachable[idx] {
                continue;
            }
            let expr_id = crate::arena::ExprId(idx as u32);
            let node = arena.node(expr_id);
            let vid = regalloc::ValueId(next_id);
            next_id += 1;
            id_map[idx] = vid;

            let map_child = |child: &crate::arena::ExprId| -> regalloc::ValueId {
                let mapped = id_map[child.0 as usize];
                assert!(
                    mapped.0 != u32::MAX,
                    "CompileWorkspace: child ExprId({}) not mapped",
                    child.0
                );
                mapped
            };

            let sched_op = match node {
                crate::arena::ExprNode::Var(i) => ScheduledOp::Var(*i),
                crate::arena::ExprNode::Const(v) => ScheduledOp::Const(*v),
                crate::arena::ExprNode::Param(i) => panic!(
                    "ExprNode::Param({}) reached CompileWorkspace — \
                     call substitute_params before compile",
                    i
                ),
                crate::arena::ExprNode::Buffer(b) => panic!(
                    "ExprNode::Buffer({}) reached CompileWorkspace — memory ops require a \
                     binding table (KERNELS_AND_LATTICES.md M2, not yet implemented)",
                    b.0
                ),
                crate::arena::ExprNode::Unary(op, child) => {
                    ScheduledOp::Unary(*op, map_child(child))
                }
                crate::arena::ExprNode::Binary(op, a, b) => {
                    ScheduledOp::Binary(*op, map_child(a), map_child(b))
                }
                crate::arena::ExprNode::Ternary(op, a, b, c) => {
                    ScheduledOp::Ternary(*op, map_child(a), map_child(b), map_child(c))
                }
                crate::arena::ExprNode::Nary(_, _, _) => {
                    panic!("Nary not supported in JIT arena compilation")
                }
            };
            schedule.push((vid, sched_op));
        }

        // 2. Build uses_map.
        self.uses_map.clear();
        for (_, op) in &self.schedule {
            let uses = match op {
                ScheduledOp::Var(_) | ScheduledOp::Const(_) => Vec::new(),
                ScheduledOp::Unary(_, a)
                | ScheduledOp::ShiftImm(_, a, _)
                | ScheduledOp::Gather(a, _) => alloc::vec![*a],
                ScheduledOp::Binary(_, a, b) => alloc::vec![*a, *b],
                ScheduledOp::Ternary(_, a, b, c) => alloc::vec![*a, *b, *c],
            };
            self.uses_map.push(uses);
        }

        // 3. Compile via the shared backend (schedule + uses_map -> executable).
        //    compile_from_schedule owns the Vecs (moved), so swap them out
        //    and restore empty Vecs with retained capacity afterward.
        let schedule_owned = core::mem::take(&mut self.schedule);
        let uses_owned = core::mem::take(&mut self.uses_map);
        let result = compile_from_schedule(
            schedule_owned,
            uses_owned,
            EmitCtx {
                max_regs: self.ctx.max_regs,
                spill_offset: 0,
                spill_count: 0,
            },
        )?;

        // Copy the compiled bytes into our reusable code buffer.
        let bytes = result.code.as_bytes();
        // SAFETY: The caller guarantees no concurrent use and that previous
        // function pointers are not held across this call.
        let func: executable::KernelFn = unsafe { self.code_buf.write_code(bytes)? };
        Ok(func)
    }
}

/// The architecture seam for the shared per-batch driver.
///
/// `compile_dag_via_backend` owns the architecture-INDEPENDENT logic — schedule,
/// register allocation, frame layout, and the Select short-circuit control flow
/// — and calls an `IsaBackend` for the leaf operations that actually differ
/// between x86-64 and aarch64 (instruction encoding, branch encoding, the
/// prologue/epilogue, and any arch-specific finalization such as aarch64's
/// constant pool). Both backends therefore run the *same* driver: there is one
/// place that decides when to emit a guard branch, where the root goes, etc.
///
/// `Branch` is an opaque per-backend fixup token (aarch64 distinguishes CBZ from
/// B; x86 uses a uniform rel32), patched later by `patch_branch`.
trait IsaBackend {
    type Branch;

    /// Number of allocatable scratch registers (the `linear_scan` budget).
    fn num_regs(&self) -> u8;

    /// Per-compile setup before any code is emitted (e.g. seed a constant pool).
    fn begin(&mut self, schedule: &[(regalloc::ValueId, ScheduledOp)]) -> Result<(), &'static str>;

    /// Called once the frame layout is known, BEFORE any body instruction is
    /// emitted. Backends whose spill addressing depends on the frame mode
    /// (x86: red zone vs allocated frame) latch it here; `prologue` runs
    /// after the body is produced and can only prepend bytes.
    fn frame_ready(&mut self, _frame_size: u32) {}

    /// Bytes actually reserved on the stack for a `frame_size`-byte *logical*
    /// spill frame.
    ///
    /// `FrameLayout` counts 16-byte slots because that is the narrowest vector
    /// any backend spills; a backend whose vector is wider reserves
    /// proportionally more in its prologue, and must report the real figure so
    /// `CompileResult::spill_bytes` describes the code that was actually
    /// emitted. Identity for the 16-byte backends (SSE2, NEON).
    fn frame_bytes(&self, frame_size: u32) -> u32 {
        frame_size
    }

    /// Function prologue (allocate the spill frame, set up any pool anchor).
    fn prologue(&mut self, code: &mut Vec<u8>, frame_size: u32);

    /// Emit one resolved instruction (with its reloads/store).
    fn emit_plan(&mut self, code: &mut Vec<u8>, plan: &InstructionPlan)
    -> Result<(), &'static str>;

    /// Register-to-register move.
    fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg);

    /// Spill a register to a frame slot.
    fn emit_store(&mut self, code: &mut Vec<u8>, src: Reg, offset: u32)
    -> Result<(), &'static str>;

    /// Resolve a value to a register, reloading/rematerializing into `target`
    /// if it is spilled or rematerialized.
    #[allow(clippy::too_many_arguments)]
    fn emit_resolve(
        &mut self,
        code: &mut Vec<u8>,
        vid: regalloc::ValueId,
        target: Reg,
        reg_for: &[Option<Reg>],
        spill_for: &[Option<u32>],
        remat_for: &[Option<u32>],
    ) -> Reg;

    /// Branch taken when `mask_reg` is all-false (skip the true arm).
    fn emit_skip_if_all_false(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> Self::Branch;
    /// Branch taken when `mask_reg` is all-true (skip the false arm).
    fn emit_skip_if_all_true(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> Self::Branch;
    /// Unconditional jump.
    fn emit_jump(&mut self, code: &mut Vec<u8>) -> Self::Branch;
    /// Patch a previously emitted branch to land at `target`.
    fn patch_branch(&mut self, code: &mut Vec<u8>, branch: Self::Branch, target: usize);

    /// Function epilogue: move the result into the return register, tear down
    /// the frame, emit RET, and perform any arch-specific finalization (e.g.
    /// append + anchor the constant pool). After this, `code` is complete.
    fn epilogue(&mut self, code: &mut Vec<u8>, result_reg: Reg, frame_size: u32);

    /// A fixed scratch register safe as a reload target for a Select whose
    /// result AND both branches are spilled (the mask occupies
    /// `RELOAD_REGS[0]` as the spilled destination, `if_true` takes
    /// `RELOAD_REGS[1]`, and `if_false` needs a fourth register). Must be
    /// outside the allocator pool and untouched by this backend's own
    /// `Select` emission.
    fn select_extra_reload(&self) -> Reg;

    /// Wrap a per-batch `body` (result in `result_reg`, spilling into a
    /// `frame_size`-slot frame) in the collapse loop scaffold, producing a
    /// complete [`CollapseKernelFn`](executable::CollapseKernelFn): the
    /// caller's lane-sequential X is an induction value stepped by the batch
    /// width each iteration, Y/Z/W are loop-invariant arguments, and each
    /// batch's result is stored straight to the output pointer. The body's
    /// branches are self-relative, so inlining it inside the loop is sound.
    ///
    /// Coordinate state lives in stack slots above the body's spill frame:
    /// the ABI's vector registers are caller-saved scratch to the body, so
    /// each iteration reloads X/Y/Z/W into the input registers from the
    /// slots and the X slot alone is stepped.
    ///
    /// `hoist` (possibly empty) is the LICM prologue: X-invariant code emitted
    /// once per call, after the coordinate stores and before the loop, parking
    /// its results in `hoist_slots` vector slots directly above the coordinate
    /// slots — the scaffold reserves them in its frame allocation.
    #[allow(clippy::too_many_arguments)] // the scaffold's full framing contract
    fn emit_collapse_loop(
        &mut self,
        hoist: &[u8],
        body: &[u8],
        result_reg: Reg,
        frame_size: u32,
        hoist_slots: u32,
    ) -> Result<Vec<u8>, &'static str>;
}

/// Drive a `(schedule, uses_map)` to machine code via an [`IsaBackend`].
///
/// This is the single shared per-batch driver for both architectures. The
/// control flow here — Select guard analysis, short-circuit branch emission and
/// patching, root resolution — is identical regardless of ISA; only the leaf
/// emits go through `backend`.
/// Emit the register-allocated *body* of a DAG — the per-op instruction stream
/// plus root resolution — with **no** prologue, epilogue, or `RET`. Returns the
/// body bytes, the register holding the root value, the spill-frame size, and
/// the spill count. The body's branches are self-relative, so a caller may
/// freely prepend a prologue / wrap it in a loop / append an epilogue. This is
/// the seam that lets both the per-batch kernel and the internal-loop collapse
/// driver share one body emitter.
fn emit_dag_body<B: IsaBackend>(
    schedule: Vec<(regalloc::ValueId, ScheduledOp)>,
    uses_map: Vec<Vec<regalloc::ValueId>>,
    backend: &mut B,
) -> Result<(Vec<u8>, Reg, u32, u32), &'static str> {
    emit_dag_body_hoisted(schedule, uses_map, backend, HoistCtx::None, None)
}

/// [`emit_dag_body`] with collapse-loop LICM support: a hoist map (see
/// [`HoistCtx`]) and an optional frame-size override. The override replaces
/// the layout's frame size in the `frame_ready` latch and the returned frame
/// size — the collapse driver passes the max of the prologue's and body's
/// frames so both address the shared hoist slots consistently (and, on x86,
/// so both latch the same allocated-frame mode).
fn emit_dag_body_hoisted<B: IsaBackend>(
    schedule: Vec<(regalloc::ValueId, ScheduledOp)>,
    uses_map: Vec<Vec<regalloc::ValueId>>,
    backend: &mut B,
    hoist: HoistCtx<'_>,
    frame_override: Option<u32>,
) -> Result<(Vec<u8>, Reg, u32, u32), &'static str> {
    use alloc::collections::BTreeMap;

    // Pre-colored values (variables -> input registers).
    let mut precolored: BTreeMap<regalloc::ValueId, Reg> = BTreeMap::new();
    for (vid, op) in &schedule {
        if let ScheduledOp::Var(i) = op {
            if (*i as usize) >= INPUT_REGS.len() {
                return Err("variable index out of range");
            }
            precolored.insert(*vid, INPUT_REGS[*i as usize]);
        }
    }

    // Register allocation (linear scan + Belady eviction + spilling).
    let mut allocation = regalloc::linear_scan(
        &schedule,
        &uses_map,
        &precolored,
        backend.num_regs(),
        SCRATCH_BASE,
    );
    let mut layout = FrameLayout::from_allocation(&allocation.spilled)?;
    let real_spill_count = layout.spill_slots.len() as u32;

    // Hoisted values in the body are pre-spilled at their hoist slots: strip
    // whatever location the allocator gave their placeholder defs and pin
    // them, so consumers reload from the slot the prologue stored to.
    if let HoistCtx::Body(hoisted) = &hoist {
        for (vid, &offset) in hoisted.iter() {
            allocation.assignment.remove(vid);
            allocation.rematerialize.remove(vid);
            layout.spill_slots.insert(*vid, offset);
        }
    }

    let frame_size = frame_override.unwrap_or(layout.frame_size);
    if frame_size < layout.frame_size {
        return Err("frame override smaller than the layout's frame");
    }
    backend.frame_ready(frame_size);

    // Select short-circuit guards (disabled in the prologue — see HoistCtx).
    let select_guards = if matches!(hoist, HoistCtx::Prologue(_)) {
        Vec::new()
    } else {
        analyze_select_guards(&schedule)
    };
    let sched_len = schedule.len();

    struct PendingBranch {
        guard_idx: usize,
        arm: u8,
    }
    let mut branch_starts: alloc::vec::Vec<alloc::vec::Vec<PendingBranch>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();
    let mut branch_ends: alloc::vec::Vec<alloc::vec::Vec<usize>> =
        (0..sched_len).map(|_| alloc::vec::Vec::new()).collect();
    for (gi, guard) in select_guards.iter().enumerate() {
        if guard.true_range.0 != guard.true_range.1 {
            branch_starts[guard.true_range.0].push(PendingBranch {
                guard_idx: gi,
                arm: 0,
            });
            if guard.true_range.1 < sched_len {
                branch_ends[guard.true_range.1].push(gi);
            }
        }
        if guard.false_range.0 != guard.false_range.1 {
            branch_starts[guard.false_range.0].push(PendingBranch {
                guard_idx: gi,
                arm: 1,
            });
            if guard.false_range.1 < sched_len {
                branch_ends[guard.false_range.1].push(gi);
            }
        }
    }

    // Dense ValueId -> location lookups for the hot loop.
    let max_vid = schedule.iter().map(|(v, _)| v.0).max().unwrap_or(0) as usize;
    let mut reg_for: alloc::vec::Vec<Option<Reg>> = alloc::vec![None; max_vid + 1];
    for (&vid, &reg) in &allocation.assignment {
        reg_for[vid.0 as usize] = Some(reg);
    }
    let mut spill_for: alloc::vec::Vec<Option<u32>> = alloc::vec![None; max_vid + 1];
    for (&vid, &offset) in &layout.spill_slots {
        spill_for[vid.0 as usize] = Some(offset);
    }
    let mut remat_for: alloc::vec::Vec<Option<u32>> = alloc::vec![None; max_vid + 1];
    for (&vid, &bits) in &allocation.rematerialize {
        remat_for[vid.0 as usize] = Some(bits);
    }

    backend.begin(&schedule)?;

    // No prologue here — the caller frames the body (see the fn doc).
    let mut code: Vec<u8> = Vec::new();

    let mut pending_patches: BTreeMap<(usize, u8), B::Branch> = BTreeMap::new();

    for (sched_idx, (vid, sched_op)) in schedule.iter().enumerate() {
        // Guard branches that begin before this instruction.
        for pb in &branch_starts[sched_idx] {
            let (guard_idx, arm) = (pb.guard_idx, pb.arm);
            let guard = &select_guards[guard_idx];
            let mask_reg = backend.emit_resolve(
                &mut code,
                guard.mask_vid,
                RELOAD_REG,
                &reg_for,
                &spill_for,
                &remat_for,
            );
            let branch = match arm {
                0 => backend.emit_skip_if_all_false(&mut code, mask_reg),
                _ => backend.emit_skip_if_all_true(&mut code, mask_reg),
            };
            pending_patches.insert((guard_idx, arm), branch);
        }

        // Guard branches that end at this instruction (patch their targets).
        for &gi in &branch_ends[sched_idx] {
            if let Some(branch) = pending_patches.remove(&(gi, 0)) {
                let target = code.len();
                backend.patch_branch(&mut code, branch, target);
            }
            if let Some(branch) = pending_patches.remove(&(gi, 1)) {
                let target = code.len();
                backend.patch_branch(&mut code, branch, target);
            }
        }

        // A hoisted value's placeholder def emits nothing — the prologue
        // already parked the value in its slot; consumers reload from there.
        if let HoistCtx::Body(hoisted) = &hoist
            && hoisted.contains_key(vid)
        {
            continue;
        }

        let dst_loc = resolve_dst_loc_dense(*vid, &reg_for, &spill_for, &remat_for);
        let plan = resolve_operands(
            sched_op,
            dst_loc,
            &allocation.assignment,
            &layout.spill_slots,
            &allocation.rematerialize,
            backend.select_extra_reload(),
        )?;

        // Select with a guard region: emit a uniform-mask short-circuit wrapper.
        if let ScheduledOp::Ternary(OpKind::Select, mask_vid, true_vid, false_vid) = sched_op
            && let Some(guard) = select_guards.iter().find(|g| g.select_idx == sched_idx)
        {
            let has_true = guard.true_range.0 != guard.true_range.1;
            let has_false = guard.false_range.0 != guard.false_range.1;
            if has_true || has_false {
                let mask_reg = backend.emit_resolve(
                    &mut code, *mask_vid, RELOAD_REG, &reg_for, &spill_for, &remat_for,
                );
                let dst = match dst_loc {
                    Loc::Reg(r) => r,
                    Loc::Spill(_) => RELOAD_REGS[0],
                };
                let true_reg = reg_for.get(true_vid.0 as usize).and_then(|r| *r);
                let false_reg = reg_for.get(false_vid.0 as usize).and_then(|r| *r);

                let all_false = backend.emit_skip_if_all_false(&mut code, mask_reg);
                let all_true = backend.emit_skip_if_all_true(&mut code, mask_reg);

                // Mixed lanes: the real select.
                backend.emit_plan(&mut code, &plan)?;
                let skip_end = backend.emit_jump(&mut code);

                // All-false: dst <- false arm.
                let all_false_target = code.len();
                if let Some(freg) = false_reg {
                    backend.emit_mov(&mut code, dst, freg);
                } else {
                    backend
                        .emit_resolve(&mut code, *false_vid, dst, &reg_for, &spill_for, &remat_for);
                }
                let skip_end2 = backend.emit_jump(&mut code);

                // All-true: dst <- true arm.
                let all_true_target = code.len();
                if let Some(treg) = true_reg {
                    backend.emit_mov(&mut code, dst, treg);
                } else {
                    backend
                        .emit_resolve(&mut code, *true_vid, dst, &reg_for, &spill_for, &remat_for);
                }

                let end_target = code.len();
                backend.patch_branch(&mut code, all_false, all_false_target);
                backend.patch_branch(&mut code, all_true, all_true_target);
                backend.patch_branch(&mut code, skip_end, end_target);
                backend.patch_branch(&mut code, skip_end2, end_target);

                if let Loc::Spill(offset) = dst_loc {
                    backend.emit_store(&mut code, dst, offset)?;
                }
                continue;
            }
        }

        backend.emit_plan(&mut code, &plan)?;

        // Prologue mode: park each hoist root in its slot right after its
        // def, while the value is guaranteed live. (Guards are disabled in
        // this mode, so every def reaches this point — the guarded-Select
        // early-continue above cannot fire.)
        if let HoistCtx::Prologue(hoisted) = &hoist
            && let Some(&offset) = hoisted.get(vid)
        {
            let r = backend.emit_resolve(
                &mut code, *vid, RELOAD_REG, &reg_for, &spill_for, &remat_for,
            );
            backend.emit_store(&mut code, r, offset)?;
        }
    }

    assert!(
        pending_patches.is_empty(),
        "BUG: {} Select short-circuit branches were never patched",
        pending_patches.len()
    );

    let root = schedule.last().map(|(v, _)| *v).expect("empty schedule");
    let result_reg = backend.emit_resolve(
        &mut code, root, RELOAD_REG, &reg_for, &spill_for, &remat_for,
    );

    Ok((code, result_reg, frame_size, real_spill_count))
}

/// Drive a `(schedule, uses_map)` to a complete per-batch function via an
/// [`IsaBackend`]: prologue, body, epilogue (root in the return register, frame
/// torn down, `RET`). Byte-for-byte identical to the pre-refactor emitter — the
/// body is simply produced by [`emit_dag_body`] and framed here.
fn compile_dag_via_backend<B: IsaBackend>(
    schedule: Vec<(regalloc::ValueId, ScheduledOp)>,
    uses_map: Vec<Vec<regalloc::ValueId>>,
    backend: &mut B,
) -> Result<CompileResult, &'static str> {
    let (body, result_reg, frame_size, spill_count) = emit_dag_body(schedule, uses_map, backend)?;

    let mut code: Vec<u8> = Vec::new();
    backend.prologue(&mut code, frame_size);
    code.extend_from_slice(&body);
    backend.epilogue(&mut code, result_reg, frame_size);

    let exec = unsafe { executable::ExecutableCode::from_code(&code)? };
    Ok(CompileResult {
        code: exec,
        spill_count,
        // The backend's own figure, not the logical one: an AVX2 slot is 32
        // bytes and an AVX-512 slot 64, so reporting `frame_size` here claimed
        // 16 bytes per spill while the prologue reserved 2x/4x that. The
        // number feeds the cost model's training metadata and frame-size
        // thresholds, so the discrepancy was silently mistraining them.
        spill_bytes: backend.frame_bytes(frame_size),
        max_regs: backend.num_regs(),
        hoisted_values: 0,
    })
}

/// A pending aarch64 branch: CBZ and B are patched differently.
#[cfg(target_arch = "aarch64")]
enum Aarch64Branch {
    Cbz(usize),
    B(usize),
}

/// aarch64 implementation of the shared driver's leaf operations.
///
/// Mechanically wraps the existing aarch64 encoders + constant pool, so the
/// emitted code is the same as the previous bespoke `compile_from_schedule`.
#[cfg(target_arch = "aarch64")]
struct Aarch64Backend {
    pool: ConstPool,
    adr_patch_pos: usize,
    max_regs: u8,
}

#[cfg(target_arch = "aarch64")]
impl Aarch64Backend {
    /// Append the constant pool after the final RET and patch the ADR anchor
    /// (upgrading to ADRP+ADD when the pool is out of ADR range). Shared by
    /// the per-batch epilogue and the collapse-loop scaffold.
    fn finish_pool(&mut self, code: &mut Vec<u8>) {
        if self.pool.is_empty() {
            return;
        }
        let adr_pos = self.adr_patch_pos;
        let estimated_offset = (code.len() as i64) - (adr_pos as i64);
        let needs_adrp = estimated_offset >= (1 << 20) - 32;
        if needs_adrp {
            code.splice(adr_pos + 4..adr_pos + 4, [0, 0, 0, 0]);
        }
        while !code.len().is_multiple_of(16) {
            code.push(0);
        }
        let pool_start = code.len();
        for &bits in &self.pool.entries {
            aarch64::emit_pool_entry(code, bits);
        }
        aarch64::patch_adr_or_adrp(code, adr_pos, pool_start, needs_adrp);
    }
}

#[cfg(target_arch = "aarch64")]
impl IsaBackend for Aarch64Backend {
    type Branch = Aarch64Branch;

    fn num_regs(&self) -> u8 {
        self.max_regs
    }

    fn begin(&mut self, schedule: &[(regalloc::ValueId, ScheduledOp)]) -> Result<(), &'static str> {
        // Seed by APPENDING into the existing pool, never replacing it: a
        // collapse compile emits two bodies through one backend (the LICM
        // prologue, then the loop body), and the prologue's bytes have the
        // first pool's X17-relative offsets baked in — resetting here left
        // them pointing into the body's rebuilt pool (wrong constants; the
        // macOS glyph-ink regression). `push_f32` dedups, and each compile
        // constructs a fresh backend, so appending is reset-equivalent for
        // single-body compiles.
        for (_, op) in schedule {
            if let ScheduledOp::Const(val) = op
                && aarch64::needs_const_pool(*val)
            {
                self.pool.push_f32(*val)?;
            }
        }
        // Builtins add up to ~60 polynomial coefficients during emission; bail
        // if the expression constants + headroom would exceed the 12-bit LDR
        // offset limit.
        const BUILTIN_HEADROOM: usize = 128;
        if self.pool.entries.len() + BUILTIN_HEADROOM > 4095 {
            return Err("expression too large: constant pool would exceed 12-bit LDR offset limit");
        }
        Ok(())
    }

    fn prologue(&mut self, code: &mut Vec<u8>, frame_size: u32) {
        if frame_size > 0 {
            aarch64::emit_sub_sp(code, frame_size);
        }
        // Builtins may add pool entries during emission, so always reserve the
        // ADR anchor (harmless if the pool ends up empty).
        self.adr_patch_pos = aarch64::emit_adr_x17_placeholder(code);
    }

    fn emit_plan(
        &mut self,
        code: &mut Vec<u8>,
        plan: &InstructionPlan,
    ) -> Result<(), &'static str> {
        emit_instruction_plan(code, plan, &mut self.pool)
    }

    fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
        emit_mov_reg(code, dst, src);
    }

    fn emit_store(
        &mut self,
        code: &mut Vec<u8>,
        src: Reg,
        offset: u32,
    ) -> Result<(), &'static str> {
        aarch64::emit_str_sp(code, src, offset);
        Ok(())
    }

    fn emit_resolve(
        &mut self,
        code: &mut Vec<u8>,
        vid: regalloc::ValueId,
        target: Reg,
        reg_for: &[Option<Reg>],
        spill_for: &[Option<u32>],
        remat_for: &[Option<u32>],
    ) -> Reg {
        emit_resolve_dense(code, vid, target, reg_for, spill_for, remat_for, &self.pool)
    }

    fn emit_skip_if_all_false(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> Aarch64Branch {
        let scratch = Reg(28);
        aarch64::emit_umaxv(code, scratch, mask_reg); // max lane; 0 => all-false
        aarch64::emit_fmov_to_gp(code, scratch);
        Aarch64Branch::Cbz(aarch64::emit_cbz_w16(code))
    }

    fn emit_skip_if_all_true(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> Aarch64Branch {
        let scratch = Reg(28);
        aarch64::emit_uminv(code, scratch, mask_reg); // min lane; 0xFFFFFFFF => all-true
        aarch64::emit_fmov_to_gp(code, scratch);
        aarch64::emit32(code, 0x2A3003F0); // MVN W16, W16  -> 0 iff all-true
        Aarch64Branch::Cbz(aarch64::emit_cbz_w16(code))
    }

    fn emit_jump(&mut self, code: &mut Vec<u8>) -> Aarch64Branch {
        Aarch64Branch::B(aarch64::emit_b(code))
    }

    fn patch_branch(&mut self, code: &mut Vec<u8>, branch: Aarch64Branch, target: usize) {
        match branch {
            Aarch64Branch::Cbz(p) => aarch64::patch_cbz_cbnz(code, p, target),
            Aarch64Branch::B(p) => aarch64::patch_b(code, p, target),
        }
    }

    fn epilogue(&mut self, code: &mut Vec<u8>, result_reg: Reg, frame_size: u32) {
        if result_reg.0 != 0 {
            emit_mov_reg(code, Reg(0), result_reg);
        }
        if frame_size > 0 {
            aarch64::emit_add_sp(code, frame_size);
        }
        // RET
        code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());
        self.finish_pool(code);
    }

    fn select_extra_reload(&self) -> Reg {
        // v28: fixed-purpose scratch outside the allocatable range; BSL
        // reads its three operands directly and never touches it.
        Reg(28)
    }

    // AAPCS64: x0 = ctx (read-only in the body's gathers), x1 = out,
    // x2 = groups, v0..3 = x0/y/z/w. Loop registers: x3 = counter; the body's
    // scratch GPRs are x9-x11 (gather), w16 (branch tests), x17 (pool anchor)
    // — all disjoint. Coordinate slots live above the body's spill frame.
    fn emit_collapse_loop(
        &mut self,
        hoist: &[u8],
        body: &[u8],
        result_reg: Reg,
        frame_size: u32,
        hoist_slots: u32,
    ) -> Result<Vec<u8>, &'static str> {
        const VW: u32 = 16;
        let total = frame_size + (4 + hoist_slots) * VW;
        let slot = |k: u32| frame_size + k * VW;
        let mut code: Vec<u8> = Vec::with_capacity(hoist.len() + body.len() + 128);

        aarch64::emit_sub_sp(&mut code, total);
        // Anchor the pool (the prologue's and body's constants load
        // X17-relative).
        self.adr_patch_pos = aarch64::emit_adr_x17_placeholder(&mut code);
        for k in 0..4u32 {
            aarch64::emit_str_sp(&mut code, Reg(k as u8), slot(k));
        }
        // LICM prologue: X-invariant values, parked in the hoist slots.
        code.extend_from_slice(hoist);
        // mov x3, xzr — batch counter.
        aarch64::emit32(&mut code, 0xD280_0003);

        let loop_top = code.len();
        // cmp x3, x2 ; b.hs end (forward, patched below).
        aarch64::emit32(&mut code, 0xEB02_007F);
        let bhs_at = code.len();
        aarch64::emit32(&mut code, 0x5400_0002);

        for k in 0..4u32 {
            aarch64::emit_ldr_sp(&mut code, Reg(k as u8), slot(k));
        }
        code.extend_from_slice(body);

        // str q(result), [x1] ; add x1, x1, #16.
        aarch64::emit32(&mut code, 0x3D80_0000 | (1 << 5) | u32::from(result_reg.0));
        aarch64::emit32(&mut code, 0x9100_4021);

        // X += lane count (4.0 is FMOV-immediate encodable; v0/v1 are about
        // to be reloaded next iteration, so they are free here).
        aarch64::emit_ldr_sp(&mut code, Reg(0), slot(0));
        aarch64::emit_fmov_imm(
            &mut code,
            Reg(1),
            VW as f32 / 4.0,
            [Reg(28), Reg(29), Reg(30), Reg(31)],
        );
        aarch64::emit_fadd(&mut code, Reg(0), Reg(0), Reg(1));
        aarch64::emit_str_sp(&mut code, Reg(0), slot(0));

        // add x3, x3, #1 ; b loop_top (backward).
        aarch64::emit32(&mut code, 0x9100_0463);
        let back = ((loop_top as i64 - code.len() as i64) / 4) as i32;
        aarch64::emit32(&mut code, 0x1400_0000 | ((back as u32) & 0x03FF_FFFF));

        // end: patch b.hs here, tear down, RET, pool.
        let end = code.len();
        let imm19 = (((end - bhs_at) / 4) as u32) & 0x7FFFF;
        let bhs = 0x5400_0000 | (imm19 << 5) | 0x2;
        code[bhs_at..bhs_at + 4].copy_from_slice(&bhs.to_le_bytes());

        aarch64::emit_add_sp(&mut code, total);
        code.extend_from_slice(&0xD65F_03C0u32.to_le_bytes());
        self.finish_pool(&mut code);
        Ok(code)
    }
}

/// Shared compilation backend: schedule + uses_map -> CompileResult.
///
/// `compile_arena_dag_with_ctx` produces the `(schedule, uses_map)` format and
/// converges on the architecture-shared [`compile_dag_via_backend`] driver via
/// [`Aarch64Backend`].
#[cfg(target_arch = "aarch64")]
fn compile_from_schedule(
    schedule: Vec<(regalloc::ValueId, ScheduledOp)>,
    uses_map: Vec<Vec<regalloc::ValueId>>,
    ctx: EmitCtx,
) -> Result<CompileResult, &'static str> {
    let mut backend = Aarch64Backend {
        pool: ConstPool::new(),
        adr_patch_pos: 0,
        max_regs: ctx.max_regs,
    };
    compile_dag_via_backend(schedule, uses_map, &mut backend)
}

/// Info about an operation in the schedule.
#[derive(Debug, Clone)]
pub enum ScheduledOp {
    /// Variable reference (input register)
    Var(u8),
    /// Constant value
    Const(f32),
    /// Unary op with input value
    Unary(OpKind, regalloc::ValueId),
    /// Binary op with input values
    Binary(OpKind, regalloc::ValueId, regalloc::ValueId),
    /// Ternary op with input values
    Ternary(
        OpKind,
        regalloc::ValueId,
        regalloc::ValueId,
        regalloc::ValueId,
    ),
    /// Bit-shift by a compile-time immediate: `op` is `Shl` or `Shr`, the value
    /// is `ValueId`, and the shift count is folded out of the `Const` RHS by
    /// `arena_to_schedule` (so it never becomes a scheduled value / register).
    ShiftImm(OpKind, regalloc::ValueId, u8),
    /// Bound-memory gather: read buffer `slot` at the lane index computed by the
    /// value operand. Lowered from `RawGather(Buffer(slot), index)`; the buffer
    /// leaf is folded out to the `slot` immediate (like `ShiftImm`'s count) so it
    /// never becomes a scheduled value. The index is the one real input.
    Gather(regalloc::ValueId, u16),
}

// =============================================================================
// Arena to Schedule (zero-cost linearization)
// =============================================================================

/// Mark nodes reachable from `root` via DFS.
///
/// The arena may contain garbage nodes from junkify passes; only nodes
/// transitively referenced by `root` should appear in the schedule.
fn mark_reachable(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
    reachable: &mut [bool],
) {
    let mut stack = alloc::vec![root];
    while let Some(id) = stack.pop() {
        let idx = id.0 as usize;
        if reachable[idx] {
            continue;
        }
        reachable[idx] = true;
        for child in arena.children(id) {
            if !reachable[child.0 as usize] {
                stack.push(child);
            }
        }
    }
}

/// Build a schedule directly from an [`ExprArena`].
///
/// The arena stores nodes in topological order (children before parents by
/// construction). We filter to reachable nodes, remap `ExprId` to `ValueId`,
/// and translate `ExprNode` to `ScheduledOp`.
///
/// # Panics
///
/// Panics if a `Param` or `Nary` node is encountered (these are not expected
/// in JIT compilation).
fn arena_to_schedule(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
) -> Vec<(regalloc::ValueId, ScheduledOp)> {
    use crate::arena::{ExprId, ExprNode};
    use regalloc::ValueId;

    let len = arena.len();
    let mut reachable = alloc::vec![false; len];
    mark_reachable(arena, root, &mut reachable);

    // ExprId to ValueId mapping. u32::MAX = unmapped (unreachable).
    let mut id_map = alloc::vec![ValueId(u32::MAX); len];
    let mut schedule = Vec::new();
    let mut next_id = 0u32;

    for idx in 0..len {
        if !reachable[idx] {
            continue;
        }
        let expr_id = ExprId(idx as u32);
        let node = arena.node(expr_id);
        let vid = ValueId(next_id);
        next_id += 1;
        id_map[idx] = vid;

        let map_child = |child: &ExprId| -> ValueId {
            let mapped = id_map[child.0 as usize];
            assert!(
                mapped.0 != u32::MAX,
                "arena_to_schedule: child ExprId({}) not yet mapped -- \
                 arena is not in topological order or child is unreachable",
                child.0
            );
            mapped
        };

        let sched_op = match node {
            ExprNode::Var(i) => ScheduledOp::Var(*i),
            ExprNode::Const(v) => ScheduledOp::Const(*v),
            ExprNode::Param(i) => panic!(
                "ExprNode::Param({}) reached the JIT emitter -- \
                 call substitute_params before compile_arena()",
                i
            ),
            // A Buffer leaf is always folded into a `Gather`'s `slot` immediate
            // (below), so any Buffer that survives as its own reachable node is a
            // dead operand — never consumed as a value. Emit a harmless dead
            // placeholder occupying its ValueId slot, exactly as ShiftImm leaves
            // its folded shift-count Const as a dead schedule entry.
            ExprNode::Buffer(_) => ScheduledOp::Const(0.0),
            ExprNode::Unary(op, child) => ScheduledOp::Unary(*op, map_child(child)),
            // Shl/Shr fold their Const shift-count operand into an immediate, so
            // the count never becomes a scheduled value (matching the imm-only
            // hardware shift encoders). The count const may still appear as its
            // own schedule entry (harmless/unused) if shared.
            ExprNode::Binary(op @ (OpKind::Shl | OpKind::Shr), a, b) => {
                let amount = match arena.node(*b) {
                    ExprNode::Const(v) => *v as u32 as u8,
                    _ => panic!(
                        "{:?} shift count must be a Const (lowering guarantees this)",
                        op
                    ),
                };
                ScheduledOp::ShiftImm(*op, map_child(a), amount)
            }
            // RawGather folds its Buffer leaf into the `slot` immediate (like a
            // shift count); only the index operand becomes a scheduled value.
            ExprNode::Binary(OpKind::RawGather, buf, idx) => {
                let slot = match arena.node(*buf) {
                    ExprNode::Buffer(id) => id.0,
                    other => panic!("RawGather's first child must be a Buffer leaf, got {other:?}"),
                };
                ScheduledOp::Gather(map_child(idx), slot)
            }
            // Unreachable precondition: every compile entry point runs
            // `lowering::lower_dwrt` before scheduling, which either rewrites
            // all `Dwrt` (autodiff) nodes into chain-rule arithmetic or errors
            // loudly on an op it cannot differentiate. A `Dwrt` here means a
            // caller bypassed that pipeline. Fail loudly rather than as a
            // cryptic instruction-emit panic.
            ExprNode::Binary(OpKind::Dwrt, _, _) => panic!(
                "arena_to_schedule: a Dwrt (autodiff) node reached the JIT \
                 emitter. lower_dwrt runs in every compile entry point and \
                 either eliminates Dwrt or refuses to compile, so a survivor \
                 means this schedule was built without the lowering pipeline."
            ),
            ExprNode::Binary(op, a, b) => ScheduledOp::Binary(*op, map_child(a), map_child(b)),
            ExprNode::Ternary(op, a, b, c) => {
                ScheduledOp::Ternary(*op, map_child(a), map_child(b), map_child(c))
            }
            ExprNode::Nary(_, _, _) => panic!("Nary not supported in JIT arena compilation"),
        };
        schedule.push((vid, sched_op));
    }
    schedule
}

/// Build uses_map from a schedule (children of each operation).
///
/// Returns a dense `Vec` indexed by `ValueId.0`: for each value, the list
/// of child `ValueId`s it consumes.
fn arena_to_uses(schedule: &[(regalloc::ValueId, ScheduledOp)]) -> Vec<Vec<regalloc::ValueId>> {
    schedule
        .iter()
        .map(|(_, op)| match op {
            ScheduledOp::Var(_) | ScheduledOp::Const(_) => Vec::new(),
            ScheduledOp::Unary(_, a) => alloc::vec![*a],
            ScheduledOp::ShiftImm(_, a, _) => alloc::vec![*a],
            ScheduledOp::Gather(a, _) => alloc::vec![*a],
            ScheduledOp::Binary(_, a, b) => alloc::vec![*a, *b],
            ScheduledOp::Ternary(_, a, b, c) => alloc::vec![*a, *b, *c],
        })
        .collect()
}

// =============================================================================
// Collapse-loop LICM (X-invariant hoisting)
// =============================================================================

/// Compute [`Variance`](crate::variance::Variance) for every schedule entry.
///
/// The schedule mirrors the arena's topological order, so one forward pass
/// suffices — the dense result is indexed by `ValueId.0`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn schedule_variance(
    schedule: &[(regalloc::ValueId, ScheduledOp)],
) -> Vec<crate::variance::Variance> {
    use crate::variance::Variance;
    let max_vid = schedule.iter().map(|(v, _)| v.0).max().unwrap_or(0) as usize;
    let mut v = alloc::vec![Variance::CONST; max_vid + 1];
    for (vid, op) in schedule {
        let i = vid.0 as usize;
        v[i] = match op {
            ScheduledOp::Var(idx) if *idx < 8 => Variance::from_var(*idx),
            ScheduledOp::Var(_) => Variance::ALL,
            ScheduledOp::Const(_) => Variance::CONST,
            ScheduledOp::Unary(_, a)
            | ScheduledOp::ShiftImm(_, a, _)
            // A gather reads from a bound buffer, whose contents are fixed for
            // the kernel's lifetime — its variance is its index's variance.
            | ScheduledOp::Gather(a, _) => v[a.0 as usize],
            ScheduledOp::Binary(_, a, b) => v[a.0 as usize].union(v[b.0 as usize]),
            ScheduledOp::Ternary(_, a, b, c) => v[a.0 as usize]
                .union(v[b.0 as usize])
                .union(v[c.0 as usize]),
        };
    }
    v
}

/// The collapse loop's LICM partition: which values leave the X loop, and the
/// two schedules that result.
///
/// `roots[i]` is parked in hoist slot `i`. `prologue` computes the roots (the
/// full X-invariant sub-DAG, original order); `body` is the loop schedule with
/// each root's entry replaced by a `Const(0.0)` placeholder — never emitted,
/// its location overridden to the hoist slot so consumers reload it through
/// the ordinary spill machinery.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
struct HoistPlan {
    roots: Vec<regalloc::ValueId>,
    prologue: Vec<(regalloc::ValueId, ScheduledOp)>,
    body: Vec<(regalloc::ValueId, ScheduledOp)>,
}

/// Partition a collapse schedule for LICM.
///
/// A hoist root is an X-invariant, non-leaf value consumed by at least one
/// X-dependent op (or the schedule root itself, when the whole kernel is
/// X-invariant — the loop degenerates to a store). `Gather`s — and anything
/// computed from one — are never hoisted: hoisting moves a value out of any
/// select-guard arm it sits in, and while speculating arithmetic is free,
/// keeping memory reads exactly where the per-batch kernel had them costs
/// nothing today (winding kernels are gather-free).
///
/// Returns `None` when nothing qualifies, leaving the caller on the plain
/// un-hoisted path.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn plan_collapse_hoist(
    schedule: &[(regalloc::ValueId, ScheduledOp)],
    variance: &[crate::variance::Variance],
) -> Option<HoistPlan> {
    use regalloc::ValueId;
    let n = schedule.len();
    if n == 0 {
        return None;
    }
    let max_vid = schedule.iter().map(|(v, _)| v.0).max().unwrap_or(0) as usize;

    let operands = |op: &ScheduledOp| -> alloc::vec::Vec<ValueId> {
        match op {
            ScheduledOp::Var(_) | ScheduledOp::Const(_) => alloc::vec![],
            ScheduledOp::Unary(_, a)
            | ScheduledOp::ShiftImm(_, a, _)
            | ScheduledOp::Gather(a, _) => {
                alloc::vec![*a]
            }
            ScheduledOp::Binary(_, a, b) => alloc::vec![*a, *b],
            ScheduledOp::Ternary(_, a, b, c) => alloc::vec![*a, *b, *c],
        }
    };

    // Which values are consumed by an X-dependent op, and which contain a
    // gather anywhere in their sub-DAG (forward pass — schedule is topological).
    let mut feeds_x_dependent = alloc::vec![false; max_vid + 1];
    let mut contains_gather = alloc::vec![false; max_vid + 1];
    for (vid, op) in schedule {
        let i = vid.0 as usize;
        let ops = operands(op);
        contains_gather[i] = matches!(op, ScheduledOp::Gather(_, _))
            || ops.iter().any(|a| contains_gather[a.0 as usize]);
        if variance[i].depends_on_x() {
            for a in &ops {
                feeds_x_dependent[a.0 as usize] = true;
            }
        }
    }

    let is_leaf = |op: &ScheduledOp| matches!(op, ScheduledOp::Var(_) | ScheduledOp::Const(_));
    let root_vid = schedule.last().map(|(v, _)| *v)?;

    let mut is_root = alloc::vec![false; max_vid + 1];
    let mut roots: Vec<ValueId> = Vec::new();
    for (vid, op) in schedule {
        let i = vid.0 as usize;
        let hoistable = variance[i].is_x_invariant()
            && !is_leaf(op)
            && !contains_gather[i]
            && (feeds_x_dependent[i] || *vid == root_vid);
        if hoistable {
            is_root[i] = true;
            roots.push(*vid);
        }
    }
    if roots.is_empty() {
        return None;
    }

    // Prologue: the transitive operand closure of the roots (all X-invariant
    // by construction), kept in original topological order.
    let mut in_prologue = alloc::vec![false; max_vid + 1];
    for r in &roots {
        in_prologue[r.0 as usize] = true;
    }
    for (vid, op) in schedule.iter().rev() {
        if in_prologue[vid.0 as usize] {
            for a in operands(op) {
                in_prologue[a.0 as usize] = true;
            }
        }
    }
    let prologue: Vec<_> = schedule
        .iter()
        .filter(|(v, _)| in_prologue[v.0 as usize])
        .cloned()
        .collect();

    // Body: backward reachability from the schedule root, treating hoist roots
    // as leaves (their entries become placeholders; operands not followed).
    let mut in_body = alloc::vec![false; max_vid + 1];
    in_body[root_vid.0 as usize] = true;
    for (vid, op) in schedule.iter().rev() {
        let i = vid.0 as usize;
        if in_body[i] && !is_root[i] {
            for a in operands(op) {
                in_body[a.0 as usize] = true;
            }
        }
    }
    let body: Vec<_> = schedule
        .iter()
        .filter(|(v, _)| in_body[v.0 as usize])
        .map(|(v, op)| {
            if is_root[v.0 as usize] {
                (*v, ScheduledOp::Const(0.0)) // placeholder; never emitted
            } else {
                (*v, op.clone())
            }
        })
        .collect();

    // Keep only roots the body actually reads (an interior invariant value
    // consumed solely by other hoisted values needs no slot). The schedule
    // root always keeps its slot — the loop stores it.
    let roots: Vec<ValueId> = roots
        .into_iter()
        .filter(|r| in_body[r.0 as usize])
        .collect();
    if roots.is_empty() {
        return None;
    }

    Some(HoistPlan {
        roots,
        prologue,
        body,
    })
}

/// The spill-frame size (16-byte logical units) a schedule will need, without
/// emitting it — `linear_scan` and `FrameLayout` are pure, so this is exactly
/// the frame `emit_dag_body` will compute for the same inputs. Used to place
/// the hoist slots above both the prologue's and the body's frames before
/// either is emitted.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn presize_frame(
    schedule: &[(regalloc::ValueId, ScheduledOp)],
    num_regs: u8,
) -> Result<u32, &'static str> {
    use alloc::collections::BTreeMap;
    let mut precolored: BTreeMap<regalloc::ValueId, Reg> = BTreeMap::new();
    for (vid, op) in schedule {
        if let ScheduledOp::Var(i) = op {
            if (*i as usize) >= INPUT_REGS.len() {
                return Err("variable index out of range");
            }
            precolored.insert(*vid, INPUT_REGS[*i as usize]);
        }
    }
    let allocation = regalloc::linear_scan(schedule, &[], &precolored, num_regs, SCRATCH_BASE);
    Ok(FrameLayout::from_allocation(&allocation.spilled)?.frame_size)
}

/// How `emit_dag_body` treats hoisted values, if any.
enum HoistCtx<'a> {
    /// No hoisting (per-batch kernels, and collapse kernels with nothing to
    /// hoist).
    None,
    /// Emitting the once-per-call prologue: after each mapped value's def,
    /// store it to its hoist slot. Select short-circuit guards are disabled —
    /// a guard could skip a hoist root's def on a uniform mask, leaving its
    /// slot garbage for the loop to read (and the prologue runs once, so the
    /// guard buys nothing).
    Prologue(&'a alloc::collections::BTreeMap<regalloc::ValueId, u32>),
    /// Emitting the loop body: mapped values are never emitted; their
    /// locations are overridden to their hoist slots so every consumer
    /// reloads through the ordinary spill machinery.
    Body(&'a alloc::collections::BTreeMap<regalloc::ValueId, u32>),
}

// =============================================================================
// Select Short-Circuit Analysis
// =============================================================================

/// Describes a Select node's short-circuit structure in the schedule.
///
/// For `Select(mask, if_true, if_false)`, identifies contiguous ranges of
/// schedule entries that are exclusive to each arm (not shared with mask
/// or the other arm). These ranges can be guarded by conditional branches.
#[derive(Debug, Clone)]
struct SelectGuard {
    /// Schedule index of the Select node itself.
    select_idx: usize,
    /// ValueId of the mask operand (already computed before arms).
    mask_vid: regalloc::ValueId,
    /// Range of schedule indices exclusive to the true arm: [true_start, true_end).
    /// Empty if true_start == true_end.
    true_range: (usize, usize),
    /// Range of schedule indices exclusive to the false arm: [false_start, false_end).
    false_range: (usize, usize),
}

/// Compute the transitive dependencies of a ValueId in the schedule.
///
/// `schedule_ops` is a dense Vec indexed by `ValueId.0`, pre-built by the
/// caller so each lookup is O(1) instead of O(n).
fn transitive_deps(
    vid: regalloc::ValueId,
    schedule_ops: &[Option<ScheduledOp>],
) -> alloc::collections::BTreeSet<regalloc::ValueId> {
    use alloc::collections::BTreeSet;

    let mut deps = BTreeSet::new();
    let mut worklist = alloc::vec![vid];
    while let Some(v) = worklist.pop() {
        if !deps.insert(v) {
            continue;
        }
        // O(1) lookup via dense Vec indexed by ValueId.0
        if let Some(Some(sop)) = schedule_ops.get(v.0 as usize) {
            match sop {
                ScheduledOp::Var(_) | ScheduledOp::Const(_) => {}
                ScheduledOp::Unary(_, c)
                | ScheduledOp::ShiftImm(_, c, _)
                | ScheduledOp::Gather(c, _) => {
                    worklist.push(*c);
                }
                ScheduledOp::Binary(_, l, r) => {
                    worklist.push(*l);
                    worklist.push(*r);
                }
                ScheduledOp::Ternary(_, a, b, c) => {
                    worklist.push(*a);
                    worklist.push(*b);
                    worklist.push(*c);
                }
            }
        }
    }
    deps
}

/// Analyze the schedule for Select nodes and compute short-circuit guard ranges.
///
/// For each Select, partitions schedule entries into:
/// - Shared: needed by mask, or by both arms (must always execute)
/// - True-exclusive: only needed by the true arm (skip if mask all-false)
/// - False-exclusive: only needed by the false arm (skip if mask all-true)
///
/// Returns guards sorted by select_idx (ascending).
fn analyze_select_guards(schedule: &[(regalloc::ValueId, ScheduledOp)]) -> Vec<SelectGuard> {
    use alloc::collections::BTreeSet;

    let mut guards = Vec::new();

    if schedule.is_empty() {
        return guards;
    }

    // Build dense lookup: schedule_ops[vid.0] = Some(&ScheduledOp) for O(1) child traversal.
    // ValueIds are sequential starting from 0 (guaranteed by arena_to_schedule).
    let max_vid = schedule.iter().map(|(v, _)| v.0).max().unwrap_or(0) as usize;
    let mut schedule_ops: alloc::vec::Vec<Option<ScheduledOp>> = alloc::vec![None; max_vid + 1];
    for (vid, sop) in schedule {
        schedule_ops[vid.0 as usize] = Some(sop.clone());
    }

    // Build dense lookup: vid_to_sched_idx[vid.0] = schedule position (u32::MAX = absent).
    let mut vid_to_sched_idx: alloc::vec::Vec<usize> = alloc::vec![usize::MAX; max_vid + 1];
    for (i, (vid, _)) in schedule.iter().enumerate() {
        vid_to_sched_idx[vid.0 as usize] = i;
    }

    // Global consumer map: consumers[v.0] = every value that reads v as an
    // operand. A node may only be guarded (skipped when its arm's mask is
    // uniform) if EVERY consumer is inside that arm's subtree (or the select
    // itself) — otherwise an outer/sibling expression reads a register the
    // branch never computed. Subtree-local exclusivity (below) is necessary but
    // NOT sufficient; this is the global check that was missing.
    let mut consumers: alloc::vec::Vec<alloc::vec::Vec<regalloc::ValueId>> =
        alloc::vec![alloc::vec::Vec::new(); max_vid + 1];
    for (vid, sop) in schedule {
        let mut add = |child: regalloc::ValueId| {
            if (child.0 as usize) <= max_vid {
                consumers[child.0 as usize].push(*vid);
            }
        };
        match sop {
            ScheduledOp::Var(_) | ScheduledOp::Const(_) => {}
            ScheduledOp::Unary(_, c)
            | ScheduledOp::ShiftImm(_, c, _)
            | ScheduledOp::Gather(c, _) => add(*c),
            ScheduledOp::Binary(_, a, b) => {
                add(*a);
                add(*b);
            }
            ScheduledOp::Ternary(_, a, b, c) => {
                add(*a);
                add(*b);
                add(*c);
            }
        }
    }

    for (i, (sel_vid, sop)) in schedule.iter().enumerate() {
        if let ScheduledOp::Ternary(OpKind::Select, mask_vid, true_vid, false_vid) = sop {
            // Compute transitive deps for each subtree using the dense O(1) lookup
            let mask_deps = transitive_deps(*mask_vid, &schedule_ops);
            let true_deps = transitive_deps(*true_vid, &schedule_ops);
            let false_deps = transitive_deps(*false_vid, &schedule_ops);

            // A node is safe to skip under this arm only if every one of its
            // consumers lies within the arm's subtree or is the select node
            // itself. Otherwise skipping it (uniform-mask short-circuit) leaves a
            // value some other expression still reads uninitialized.
            let only_used_within = |v: regalloc::ValueId, arm: &BTreeSet<regalloc::ValueId>| {
                consumers[v.0 as usize]
                    .iter()
                    .all(|c| *c == *sel_vid || arm.contains(c))
            };

            // True-exclusive: in true_deps but NOT in mask_deps and NOT in
            // false_deps, AND used only within the true arm.
            let true_exclusive: BTreeSet<regalloc::ValueId> = true_deps
                .difference(&mask_deps)
                .copied()
                .collect::<BTreeSet<_>>()
                .difference(&false_deps)
                .copied()
                .filter(|v| only_used_within(*v, &true_deps))
                .collect();

            // False-exclusive: symmetric.
            let false_exclusive: BTreeSet<regalloc::ValueId> = false_deps
                .difference(&mask_deps)
                .copied()
                .collect::<BTreeSet<_>>()
                .difference(&true_deps)
                .copied()
                .filter(|v| only_used_within(*v, &false_deps))
                .collect();

            // Map to schedule indices using dense O(1) lookup
            let true_indices: BTreeSet<usize> = true_exclusive
                .iter()
                .filter_map(|v| {
                    let idx = *vid_to_sched_idx.get(v.0 as usize)?;
                    if idx == usize::MAX { None } else { Some(idx) }
                })
                .collect();
            let false_indices: BTreeSet<usize> = false_exclusive
                .iter()
                .filter_map(|v| {
                    let idx = *vid_to_sched_idx.get(v.0 as usize)?;
                    if idx == usize::MAX { None } else { Some(idx) }
                })
                .collect();

            // The guard's uniformity test is emitted at the range START and
            // reads the mask's register there, so the mask must already be
            // computed by then. Schedules from the macro pipeline emit the mask
            // before both arms, but arena-composed kernels (`Kernel::select`
            // splicing arbitrary fragments) may schedule an arm BEFORE the
            // mask — guarding such an arm would branch on an uninitialized
            // register. No guard in that case; the select still evaluates
            // correctly through the unconditional BSL/blend path.
            let mask_idx = vid_to_sched_idx
                .get(mask_vid.0 as usize)
                .copied()
                .unwrap_or(usize::MAX);

            // Get contiguous ranges (min..max+1)
            let true_range = if true_indices.is_empty() {
                (i, i) // empty range
            } else {
                let start = *true_indices
                    .iter()
                    .next()
                    .expect("non-empty set has first element");
                let end = *true_indices
                    .iter()
                    .next_back()
                    .expect("non-empty set has last element")
                    + 1;
                // The branch skips the WHOLE range [start, end) when the mask is
                // uniform, so EVERY index in it must be a true-exclusive node.
                // If any in-range index is a shared node (used outside this arm)
                // or a false-exclusive node, skipping it would leave a value some
                // other expression reads uninitialized — fall back to BSL.
                let all_exclusive = (start..end).all(|idx| true_indices.contains(&idx));
                if all_exclusive && mask_idx < start {
                    (start, end)
                } else {
                    (i, i)
                }
            };

            let false_range = if false_indices.is_empty() {
                (i, i)
            } else {
                let start = *false_indices
                    .iter()
                    .next()
                    .expect("non-empty set has first element");
                let end = *false_indices
                    .iter()
                    .next_back()
                    .expect("non-empty set has last element")
                    + 1;
                let all_exclusive = (start..end).all(|idx| false_indices.contains(&idx));
                if all_exclusive && mask_idx < start {
                    (start, end)
                } else {
                    (i, i)
                }
            };

            // Only create a guard if at least one arm has exclusive nodes
            if true_range.0 != true_range.1 || false_range.0 != false_range.1 {
                guards.push(SelectGuard {
                    select_idx: i,
                    mask_vid: *mask_vid,
                    true_range,
                    false_range,
                });
            }
        }
    }

    guards
}

/// Emit MOV (vector register copy) — used by emit_instruction_plan.
#[cfg(target_arch = "aarch64")]
fn emit_mov_reg(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    if dst.0 != src.0 {
        // ORR Vd.16B, Vn.16B, Vn.16B
        let inst = 0x4EA01C00u32 | (dst.0 as u32) | ((src.0 as u32) << 5) | ((src.0 as u32) << 16);
        code.extend_from_slice(&inst.to_le_bytes());
    }
}

/// Resolve a scheduled operation into a concrete instruction plan.
///
/// This is a PURE FUNCTION: no mutation, no side effects, no code emission.
/// Given the scheduled op, destination location, register assignments, and
/// spill slots, it computes exactly which registers to use and what
/// reload/store instructions are needed.
///
/// The two reload registers (RELOAD_REGS[0], RELOAD_REGS[1]) are used as:
///   - RELOAD_REGS[0] (v26): destination when dst is spilled, also temporary
///   - RELOAD_REGS[1] (v27): operand reload, never aliases dst
///
/// The "dst-as-temporary" trick: ARM NEON reads all sources before writing
/// the destination, so loading a spilled operand into dst is safe for binary
/// ops (the instruction reads before it writes).
#[allow(clippy::too_many_arguments)] // the register-allocation state is one logical argument
pub fn resolve_operands(
    op: &ScheduledOp,
    dst_loc: Loc,
    assignment: &alloc::collections::BTreeMap<regalloc::ValueId, Reg>,
    spill_slots: &alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    rematerialize: &alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    select_extra: Reg,
) -> Result<InstructionPlan, &'static str> {
    let tmp_op = RELOAD_REGS[1]; // v27 — always safe for operand reload

    // Compute destination: a real register, or RELOAD_REGS[0] (v26) if spilled.
    let dst = match dst_loc {
        Loc::Reg(r) => r,
        Loc::Spill(_) => RELOAD_REGS[0],
    };

    let mut reloads = Vec::new();
    let mut setup_mov = None;

    // Resolve a value to its register, or plan a reload from stack/constant into `target`.
    let resolve = |v: regalloc::ValueId, target: Reg, reloads: &mut Vec<Reload>| -> Reg {
        if let Some(&reg) = assignment.get(&v) {
            reg
        } else if let Some(&bits) = rematerialize.get(&v) {
            reloads.push(Reload::Const {
                target,
                val_bits: bits,
            });
            target
        } else if let Some(&offset) = spill_slots.get(&v) {
            reloads.push(Reload::FromStack { target, offset });
            target
        } else {
            panic!(
                "value {:?} not found in assignment, spill slots, or rematerialize",
                v
            );
        }
    };

    let resolved_op = match op {
        ScheduledOp::Var(_) => {
            // Precolored to input register — no code needed.
            ResolvedOp::Nop
        }
        ScheduledOp::Const(val) => ResolvedOp::LoadConst {
            dst,
            val_bits: val.to_bits(),
        },
        ScheduledOp::Unary(op_kind, child) => {
            let src = resolve(*child, tmp_op, &mut reloads);
            ResolvedOp::Unary {
                op: *op_kind,
                dst,
                src,
            }
        }
        ScheduledOp::ShiftImm(op_kind, child, amount) => {
            let src = resolve(*child, tmp_op, &mut reloads);
            ResolvedOp::ShiftImm {
                op: *op_kind,
                dst,
                src,
                amount: *amount,
            }
        }
        ScheduledOp::Gather(child, slot) => {
            let idx = resolve(*child, tmp_op, &mut reloads);
            ResolvedOp::Gather {
                dst,
                idx,
                slot: *slot,
            }
        }
        ScheduledOp::Binary(op_kind, left, right) => {
            let l_spilled = !assignment.contains_key(left);
            let r_spilled = !assignment.contains_key(right);

            let (l_reg, r_reg) = match (l_spilled, r_spilled) {
                (false, false) => (assignment[left], assignment[right]),
                (true, false) => {
                    let l = resolve(*left, tmp_op, &mut reloads);
                    (l, assignment[right])
                }
                (false, true) => {
                    let r = resolve(*right, tmp_op, &mut reloads);
                    (assignment[left], r)
                }
                (true, true) => {
                    // Both spilled. Left → dst (temporary), right → tmp_op.
                    let l = resolve(*left, dst, &mut reloads);
                    let r = resolve(*right, tmp_op, &mut reloads);
                    (l, r)
                }
            };
            ResolvedOp::Binary {
                op: *op_kind,
                dst,
                left: l_reg,
                right: r_reg,
            }
        }
        ScheduledOp::Ternary(op_kind, a, b, c) => {
            let a_spilled = !assignment.contains_key(a);
            let b_spilled = !assignment.contains_key(b);

            match op_kind {
                OpKind::MulAdd => {
                    // MulAdd(a, b, c) = a*b + c.
                    if a_spilled && b_spilled {
                        // Decompose: FMUL(dst, a, b) then FADD(dst, dst, c).
                        // a → dst (temp), b → tmp_op loaded upfront.
                        // c is loaded AFTER FMUL (may reuse tmp_op).
                        let a_reg = resolve(*a, dst, &mut reloads);
                        let b_reg = resolve(*b, tmp_op, &mut reloads);
                        // c is deferred — don't add to upfront reloads.
                        let (c_reg, c_deferred) = if let Some(&reg) = assignment.get(c) {
                            (reg, None)
                        } else if let Some(&bits) = rematerialize.get(c) {
                            (tmp_op, Some(DeferredReload::Const(bits)))
                        } else if let Some(&offset) = spill_slots.get(c) {
                            (tmp_op, Some(DeferredReload::FromStack(offset)))
                        } else {
                            panic!(
                                "value {:?} not found in assignment, spill slots, or rematerialize",
                                c
                            );
                        };
                        ResolvedOp::DecomposedMulAdd {
                            dst,
                            a: a_reg,
                            b: b_reg,
                            c: c_reg,
                            c_deferred,
                        }
                    } else {
                        // FMLA path: dst += a * b, so dst must hold c first.
                        // At most 1 of {a, b} is spilled → tmp_op handles it.
                        let c_reg = resolve(*c, dst, &mut reloads);
                        if dst.0 != c_reg.0 {
                            setup_mov = Some((dst, c_reg));
                        }
                        let a_reg = resolve(*a, tmp_op, &mut reloads);
                        let b_reg = resolve(*b, tmp_op, &mut reloads);
                        ResolvedOp::FusedMulAdd {
                            dst,
                            a: a_reg,
                            b: b_reg,
                        }
                    }
                }
                OpKind::Select => {
                    // BSL/blend is a 3-input RMW: the mask must end up in `dst`,
                    // and if_true / if_false each need their own live register.
                    //
                    // Reload a spilled mask STRAIGHT into `dst` — never via
                    // tmp_op. Reloads all emit before `setup_mov`, so routing the
                    // mask through tmp_op while a spilled branch also reloads
                    // into tmp_op would overwrite the mask before it reaches
                    // `dst`. Going straight to `dst` also frees tmp_op and
                    // RELOAD_REGS[0] as two independent slots for the branches.
                    let c_spilled = !assignment.contains_key(c);
                    let a_reg = resolve(*a, dst, &mut reloads);
                    if dst.0 != a_reg.0 {
                        setup_mov = Some((dst, a_reg));
                    }
                    let b_reg = resolve(*b, tmp_op, &mut reloads);
                    let c_reg = if b_spilled && c_spilled {
                        // Both branches spilled → two DISTINCT reload targets.
                        // tmp_op (RELOAD_REGS[1]) holds if_true; if_false takes
                        // RELOAD_REGS[0] — free whenever `dst` is a real
                        // register (reload regs are never in the allocator
                        // pool). A spilled *result* (dst == RELOAD_REGS[0],
                        // holding the mask) needs a fourth register: the
                        // backend's `select_extra_reload`, a fixed scratch its
                        // own Select emission never touches. Optimized glyph
                        // arenas reach this under real register pressure.
                        if dst.0 == RELOAD_REGS[0].0 {
                            resolve(*c, select_extra, &mut reloads)
                        } else {
                            resolve(*c, RELOAD_REGS[0], &mut reloads)
                        }
                    } else {
                        // At most one branch spilled → tmp_op suffices for it.
                        resolve(*c, tmp_op, &mut reloads)
                    };
                    ResolvedOp::Select {
                        dst,
                        if_true: b_reg,
                        if_false: c_reg,
                    }
                }
                _ => return Err("unsupported ternary op in DAG compilation"),
            }
        }
    };

    // Store result if destination is spilled.
    let store = if let Loc::Spill(offset) = dst_loc {
        Some(Store { src: dst, offset })
    } else {
        None
    };

    Ok(InstructionPlan {
        reloads,
        op: resolved_op,
        setup_mov,
        store,
    })
}

/// Emit machine code for a resolved instruction plan.
///
/// This is a DETERMINISTIC DISPATCH: given a plan, emit the exact
/// instructions. No decisions are made here — all decisions were
/// made by resolve_operands.
#[cfg(target_arch = "aarch64")]
fn emit_instruction_plan(
    code: &mut Vec<u8>,
    plan: &InstructionPlan,
    pool: &mut ConstPool,
) -> Result<(), &'static str> {
    use aarch64::*;

    // 1. Emit reloads (from stack or rematerialized constants)
    for reload in &plan.reloads {
        match reload {
            Reload::FromStack { target, offset } => {
                emit_ldr_sp(code, *target, *offset);
            }
            Reload::Const { target, val_bits } => {
                emit_const_load(code, *target, *val_bits, pool);
            }
        }
    }

    // 2. Emit setup MOV (for FMLA accumulator or BSL mask)
    if let Some((dst, src)) = plan.setup_mov {
        emit_mov_reg(code, dst, src);
    }

    // 3. Emit main op
    match &plan.op {
        ResolvedOp::Nop => {}
        ResolvedOp::LoadConst { dst, val_bits } => {
            emit_const_load(code, *dst, *val_bits, pool);
        }
        ResolvedOp::Unary { op, dst, src } => {
            let scratch = [Reg(28), Reg(29), Reg(30), Reg(31)];
            emit_unary(code, pool, *op, *dst, *src, scratch)?;
        }
        ResolvedOp::ShiftImm {
            op,
            dst,
            src,
            amount,
        } => {
            aarch64::emit_shift_imm(code, *op, *dst, *src, *amount)?;
        }
        ResolvedOp::Gather { dst, idx, slot } => {
            // dst = buffer[slot][idx], via four scalar loads (NEON has no
            // native gather). The context pointer (array of buffer base
            // pointers) is caller-provided in x0 per AAPCS64 — disjoint from
            // the coordinate vectors in v0..3 and never touched by the
            // arithmetic/const emit, so it survives to here.
            // v28 is fixed-purpose scratch (outside the allocatable range);
            // x9-x11 are caller-saved GPR scratch clear of the branch guard
            // (w16) and the const-pool anchor (x17).
            const IDX_INT: Reg = Reg(28);
            const BASE_GPR: u8 = 9;
            const IDX_GPR: u8 = 10;
            const VAL_GPR: u8 = 11;
            const CTX_GPR: u8 = 0;
            emit_fcvtzs(code, IDX_INT, *idx); // float idx -> int32 lanes
            emit_ldr_x_imm(code, BASE_GPR, CTX_GPR, (*slot as u32) * 8);
            emit_gather(
                code,
                *dst,
                IDX_INT,
                aarch64::GatherGprs {
                    base: BASE_GPR,
                    idx: IDX_GPR,
                    val: VAL_GPR,
                },
            );
        }
        ResolvedOp::Binary {
            op,
            dst,
            left,
            right,
        } => {
            // Every transcendental is expanded to arithmetic by
            // `expand_transcendentals` before codegen, so only primitives
            // reach here.
            emit_binary(code, *op, *dst, *left, *right);
        }
        ResolvedOp::FusedMulAdd { dst, a, b } => {
            // setup_mov already placed c into dst
            emit_fmla(code, *dst, *a, *b);
        }
        ResolvedOp::DecomposedMulAdd {
            dst,
            a,
            b,
            c,
            c_deferred,
        } => {
            // FMUL(dst, a, b) — consumes a and b (loaded upfront).
            emit_fmul(code, *dst, *a, *b);
            // Reload c after FMUL (c may reuse tmp_op which held b).
            emit_deferred(code, *c, c_deferred.as_ref(), pool);
            // FADD(dst, dst, c)
            emit_fadd(code, *dst, *dst, *c);
        }
        ResolvedOp::Select {
            dst,
            if_true,
            if_false,
        } => {
            // setup_mov already placed mask into dst
            emit_bsl(code, *dst, *if_true, *if_false);
        }
    }

    // 4. Emit store
    if let Some(store) = &plan.store {
        emit_str_sp(code, store.src, store.offset);
    }

    Ok(())
}

/// Emit a deferred reload: either from stack or rematerialized constant.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn emit_deferred(
    code: &mut Vec<u8>,
    target: Reg,
    deferred: Option<&DeferredReload>,
    pool: &ConstPool,
) {
    match deferred {
        Some(DeferredReload::FromStack(offset)) => {
            aarch64::emit_ldr_sp(code, target, *offset);
        }
        Some(DeferredReload::Const(val_bits)) => {
            emit_const_load(code, target, *val_bits, pool);
        }
        None => {}
    }
}

/// Dense-Vec variant of `resolve_dst_loc` for use in the hot emit loop.
///
/// Accepts pre-built `Vec<Option<T>>` slices indexed by `ValueId.0` for O(1)
/// lookups instead of O(log n) BTreeMap lookups.
fn resolve_dst_loc_dense(
    vid: regalloc::ValueId,
    reg_for: &[Option<Reg>],
    spill_for: &[Option<u32>],
    remat_for: &[Option<u32>],
) -> Loc {
    let idx = vid.0 as usize;
    if let Some(Some(reg)) = reg_for.get(idx) {
        Loc::Reg(*reg)
    } else if let Some(Some(offset)) = spill_for.get(idx) {
        Loc::Spill(*offset)
    } else if matches!(remat_for.get(idx), Some(Some(_))) {
        // Rematerialized values don't need a destination — they're constants
        // that will be re-emitted on each use. The "definition" is a no-op.
        // Use RELOAD_REGS[0] as a dummy — won't be written.
        Loc::Reg(RELOAD_REGS[0])
    } else {
        panic!(
            "value {:?} has no assignment, spill slot, or rematerialize entry",
            vid
        );
    }
}

/// Dense-Vec variant of `emit_resolve` for use in the hot emit loop.
///
/// Accepts pre-built `Vec<Option<T>>` slices indexed by `ValueId.0` for O(1)
/// lookups instead of O(log n) BTreeMap lookups.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn emit_resolve_dense(
    code: &mut Vec<u8>,
    vid: regalloc::ValueId,
    target: Reg,
    reg_for: &[Option<Reg>],
    spill_for: &[Option<u32>],
    remat_for: &[Option<u32>],
    pool: &ConstPool,
) -> Reg {
    let idx = vid.0 as usize;
    if let Some(Some(reg)) = reg_for.get(idx) {
        *reg
    } else if let Some(Some(bits)) = remat_for.get(idx) {
        emit_const_load(code, target, *bits, pool);
        target
    } else if let Some(Some(offset)) = spill_for.get(idx) {
        aarch64::emit_ldr_sp(code, target, *offset);
        target
    } else {
        panic!(
            "value {:?} has no register, spill slot, or rematerialize entry",
            vid
        );
    }
}

/// Compile an [`ExprArena`] DAG to executable code (x86-64).
#[cfg(target_arch = "x86_64")]
pub fn compile_arena(
    arena: &ExprArena,
    root: ExprId,
) -> Result<executable::ExecutableCode, &'static str> {
    compile_arena_dag(arena, root).map(|r| r.code)
}

/// Compile an [`ExprArena`] DAG to executable code (x86-64).
///
/// Walks the arena directly with a Sethi-Ullman tree emitter — no intermediate
/// `Expr` tree is materialized. Expressions produced by the extraction pipeline
/// are trees (no DAG sharing), so tree-walk emission is exact; shared nodes are
/// simply re-emitted.
#[cfg(target_arch = "x86_64")]
pub fn compile_arena_dag(arena: &ExprArena, root: ExprId) -> Result<CompileResult, &'static str> {
    compile_arena_dag_with_ctx(arena, root, EmitCtx::default())
}

/// Compile an [`ExprArena`] DAG with an explicit register budget (x86-64).
///
/// The `ctx` budget is advisory on x86-64: the Sethi-Ullman emitter assigns
/// scratch registers by tree position rather than performing graph-coloring
/// allocation, so it never spills within the supported depth bound.
#[cfg(target_arch = "x86_64")]
pub fn compile_arena_dag_with_ctx(
    arena: &ExprArena,
    root: ExprId,
    _ctx: EmitCtx,
) -> Result<CompileResult, &'static str> {
    // Per-batch x86 runs the architecture-shared driver
    // (`compile_dag_via_backend`): schedule -> regalloc (with spilling) -> Select
    // guards -> emit.
    //
    // Backend = build width: AVX-512 (512-bit zmm) when compiled +avx512f, else
    // SSE2 (128-bit xmm). Both implement `IsaBackend`, so the driver and the
    // KernelFn ABI stay consistent with the selected width.
    //
    // Lower, in order: derivatives (Dwrt -> chain-rule arithmetic), reductions
    // (unroll -> arithmetic + gathers), then memory reads (Gather -> index math
    // + RawGather), then transcendentals. Each is a no-op when absent. Dwrt runs
    // first because its rules emit Sin/Cos/Exp nodes the transcendental pass
    // must still lower; Reduce runs before Gather so the gathers it unrolls get
    // lowered too. A kernel that reads bound memory takes its buffer bases from
    // a context pointer in rdi; the emitted body is identical either way, so the
    // caller invokes it as `CtxKernelFn` instead of `KernelFn`.
    let (arena, root) = lowering::lower_dwrt_owned(arena, root)?;
    let (arena, root) = lowering::expand_reduce_owned(&arena, root);
    let (arena, root) = lowering::expand_gather_owned(&arena, root);
    let (arena, root) = lowering::expand_transcendentals_owned(&arena, root);
    let arena = &arena;
    let schedule = arena_to_schedule(arena, root);
    let uses = arena_to_uses(&schedule);
    // Three-way build-width split, matching `pixelflow_ir::JIT_VECTOR_BYTES`
    // and `pixelflow-core`'s `Field` storage: AVX-512 > AVX2 > SSE2.
    #[cfg(target_feature = "avx512f")]
    {
        compile_dag_via_backend(schedule, uses, &mut Avx512Backend)
    }
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    {
        compile_dag_via_backend(schedule, uses, &mut Avx2Backend)
    }
    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
    {
        compile_dag_via_backend(schedule, uses, &mut X86Backend::default())
    }
}

// =============================================================================
// x86-64 IsaBackend: the leaf emit for the SHARED per-batch driver.
// =============================================================================
//
// `compile_dag_via_backend` owns the control flow (schedule, regalloc, Select
// short-circuit guards, root handling) for BOTH architectures; this is just the
// x86 instruction encoding behind that seam, so x86 and aarch64 run the same
// driver (no "works on my machine" divergence between them).
//
// Spills go to the System V red zone when the frame fits (the kernel is a
// leaf): a `FrameLayout` slot at byte `offset` maps to `[rsp - (offset + 16)]`.
// Larger frames (glyph-scale fused kernels) allocate a real frame in the
// prologue (`sub rsp, N`); slots then live at `[rsp + offset]` and the
// epilogue releases the frame before `ret`.
//
// Register roles: xmm0-3 inputs (precolored), xmm4-9 allocatable (6),
// xmm10 fixed scratch (binary two-operand hazard + select temp), xmm11-12
// reload (`RELOAD_REGS`), xmm13-15 builtin scratch.

/// Allocatable scratch register count handed to `linear_scan` on x86 (xmm4-9).
#[cfg(target_arch = "x86_64")]
const X86_SCHED_NUM_REGS: u8 = 6;

/// Allocatable scratch register count handed to `linear_scan` on AVX2 (ymm4-7).
///
/// Two fewer than SSE2's six. AVX2's gather splits into 128-bit halves and so
/// needs two scratch registers beyond the pair SSE2 uses (ymm13/14) to hold the
/// high-half indices and the high-half result across the recombine. Those live
/// in ymm8/ymm9, which must therefore sit OUTSIDE the allocator's range: with
/// six allocatable (ymm4-9) the allocator could hand `dst` or `idx` an ymm8/9
/// that the gather then overwrites mid-sequence, silently returning wrong
/// lanes — reachable whenever five values stay live across a gather.
///
/// The cost is more spilling in AVX2 kernels generally, to fix a bug on the
/// gather path specifically. Spilling the two half-temporaries to the red zone
/// instead would restore the sixth register; that is a contained change to
/// `avx2::emit_gather_scalar` and is the better long-term fix.
#[cfg(target_arch = "x86_64")]
const AVX2_SCHED_NUM_REGS: u8 = 4;

/// Fixed scratch outside the allocatable range / reload regs: used for the
/// binary two-operand hazard and as the select blend temp.
///
/// SSE2-only (like the rest of this section down to `X86Backend`): dead under
/// a build that selects `Avx2Backend`/`Avx512Backend` instead, so gated off
/// those widths — otherwise it (and `X86Backend`) would be unreachable dead
/// code under `+avx2`/`+avx512f`, which the test-matrix's per-ISA clippy pass
/// would then flag as a *new* failure on those levels only.
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
const X86_SCRATCH: Reg = Reg(10);

/// Scratch quad for builtins (sin/cos/exp/atan2/...), which need FOUR distinct
/// scratch registers. Clear of the allocatable range (4-9) and the reload regs
/// (11,12). Includes `X86_SCRATCH` (xmm10): builtins don't use it for the
/// hazard/select roles, so it is free as a fourth scratch here.
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
const X86_BUILTIN_SCRATCH: [Reg; 4] = [Reg(10), Reg(13), Reg(14), Reg(15)];

/// Map a `FrameLayout` spill offset to a red-zone `[rsp+disp8]` displacement.
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
fn x86_redzone_disp(offset: u32) -> Result<i8, &'static str> {
    // Slots live below rsp: offset 0 -> [rsp-16], 16 -> [rsp-32], ...
    // Only called in red-zone mode (the prologue switches to an allocated
    // frame when the layout exceeds the zone), so overflow here is an
    // internal invariant violation, not a kernel-size limit.
    let disp = -(offset as i64 + 16);
    if disp < -128 {
        return Err("x86 spill: red-zone displacement out of range (prologue mode bug)");
    }
    Ok(disp as i8)
}

/// `dst = left op right` honoring SSE's two-operand form for *any* register
/// assignment from the allocator.
///
/// `emit_binary` computes `dst <- left; dst op= right`, which corrupts `right`
/// when `dst == right` and `dst != left`. The allocator may assign `dst ==
/// right`, so handle it: swap for commutative ops, otherwise stash `right` in
/// the fixed scratch.
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
fn emit_binary_safe(code: &mut Vec<u8>, op: OpKind, dst: Reg, left: Reg, right: Reg) {
    use x86_64::*;
    let commutative = matches!(
        op,
        OpKind::Add | OpKind::Mul | OpKind::Min | OpKind::Max | OpKind::Eq | OpKind::Ne
    );
    if dst == left || dst != right {
        emit_binary(code, op, dst, left, right);
    } else if commutative {
        emit_binary(code, op, dst, right, left);
    } else {
        emit_movaps(code, X86_SCRATCH, right);
        emit_movaps(code, dst, left);
        emit_binary(code, op, dst, dst, X86_SCRATCH);
    }
}

/// x86-64 implementation of the shared driver's leaf operations.
///
/// `frame_bytes` is set by `prologue`: 0 = spills fit the 128-byte red zone
/// (no frame is allocated), otherwise the size of the allocated frame and
/// spill slots are `[rsp + offset]`.
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
#[derive(Default)]
struct X86Backend {
    frame_bytes: u32,
}

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
impl X86Backend {
    fn spill_store(&self, code: &mut Vec<u8>, src: Reg, offset: u32) {
        if self.frame_bytes == 0 {
            let disp = x86_redzone_disp(offset).expect("red-zone mode implies fitting offsets");
            x86_64::emit_movups_store_rsp(code, src, disp);
        } else {
            x86_64::emit_movups_store_rsp32(code, src, offset as i32);
        }
    }

    fn spill_load(&self, code: &mut Vec<u8>, dst: Reg, offset: u32) {
        if self.frame_bytes == 0 {
            let disp = x86_redzone_disp(offset).expect("red-zone mode implies fitting offsets");
            x86_64::emit_movups_load_rsp(code, dst, disp);
        } else {
            x86_64::emit_movups_load_rsp32(code, dst, offset as i32);
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
impl IsaBackend for X86Backend {
    /// rel32 field offset of the branch (uniform for jcc/jmp on x86).
    type Branch = usize;

    fn num_regs(&self) -> u8 {
        X86_SCHED_NUM_REGS
    }

    fn begin(
        &mut self,
        _schedule: &[(regalloc::ValueId, ScheduledOp)],
    ) -> Result<(), &'static str> {
        Ok(()) // x86 const loads are self-contained; no pool.
    }

    fn frame_ready(&mut self, frame_size: u32) {
        // Red zone when it fits (max slot offset frame_size-16 maps to disp
        // -(frame_size) >= -128); otherwise an allocated frame, with slots at
        // [rsp + offset]. Latched here so the body's spill addressing agrees
        // with the prologue emitted afterwards.
        self.frame_bytes = if frame_size <= 128 { 0 } else { frame_size };
    }

    fn prologue(&mut self, code: &mut Vec<u8>, _frame_size: u32) {
        // movups is alignment-agnostic, so the frame only needs its
        // 16-multiple size; red-zone mode needs no setup for a leaf.
        if self.frame_bytes > 0 {
            x86_64::emit_sub_rsp(code, self.frame_bytes);
        }
    }

    fn emit_plan(
        &mut self,
        code: &mut Vec<u8>,
        plan: &InstructionPlan,
    ) -> Result<(), &'static str> {
        use x86_64::*;
        for reload in &plan.reloads {
            match reload {
                Reload::FromStack { target, offset } => {
                    self.spill_load(code, *target, *offset);
                }
                Reload::Const { target, val_bits } => {
                    emit_const(
                        code,
                        *target,
                        f32::from_bits(*val_bits),
                        X86_BUILTIN_SCRATCH,
                    );
                }
            }
        }
        if let Some((dst, src)) = plan.setup_mov {
            emit_movaps(code, dst, src);
        }
        match &plan.op {
            ResolvedOp::Nop => {}
            ResolvedOp::LoadConst { dst, val_bits } => {
                emit_const(code, *dst, f32::from_bits(*val_bits), X86_BUILTIN_SCRATCH);
            }
            ResolvedOp::Unary { op, dst, src } => {
                emit_unary(code, *op, *dst, *src, X86_BUILTIN_SCRATCH);
            }
            ResolvedOp::ShiftImm {
                op,
                dst,
                src,
                amount,
            } => {
                emit_shift_imm(code, *op, *dst, *src, *amount);
            }
            ResolvedOp::Gather { dst, idx, slot } => {
                // No AVX2 here, so no 128-bit vgatherdps: assemble the lanes
                // from four scalar loads. The context pointer (array of buffer
                // base pointers) is caller-provided in rdi and never touched by
                // arithmetic/const emission, so it survives to here; rax/rcx
                // are caller-saved and unused by the rest of the body.
                // Both vector scratch registers are outside the allocatable
                // range and the reload pair (see X86_BUILTIN_SCRATCH).
                x86_64::emit_gather_scalar(
                    code,
                    *dst,
                    *idx,
                    *slot,
                    x86_64::GatherScratch {
                        base_gpr: 0,  // rax
                        index_gpr: 1, // rcx
                        ctx_gpr: 7,   // rdi
                        idx_lanes: Reg(13),
                        value: Reg(14),
                    },
                );
            }
            ResolvedOp::Binary {
                op,
                dst,
                left,
                right,
            } => emit_binary_safe(code, *op, *dst, *left, *right),
            ResolvedOp::Select {
                dst,
                if_true,
                if_false,
            } => {
                // setup_mov already placed the mask in `dst`; blend in place.
                emit_select(code, *dst, *dst, *if_true, *if_false, X86_SCRATCH);
            }
            ResolvedOp::FusedMulAdd { dst, a, b } => {
                // No hardware FMA assumed: `dst` already holds c (setup_mov);
                // compute a*b in the fixed scratch, then add. a,b are never
                // X86_SCRATCH (allocator/reload regs), and `a` is copied out
                // before any write, so c==a / c==b are handled.
                emit_movaps(code, X86_SCRATCH, *a);
                emit_binary(code, OpKind::Mul, X86_SCRATCH, X86_SCRATCH, *b);
                emit_binary(code, OpKind::Add, *dst, *dst, X86_SCRATCH);
            }
            ResolvedOp::DecomposedMulAdd {
                dst,
                a,
                b,
                c,
                c_deferred,
            } => {
                // dst = a*b, reload c (after the multiply, if deferred), dst += c.
                emit_binary_safe(code, OpKind::Mul, *dst, *a, *b);
                match c_deferred {
                    Some(DeferredReload::FromStack(off)) => {
                        self.spill_load(code, *c, *off);
                    }
                    Some(DeferredReload::Const(bits)) => {
                        emit_const(code, *c, f32::from_bits(*bits), X86_BUILTIN_SCRATCH);
                    }
                    None => {}
                }
                emit_binary_safe(code, OpKind::Add, *dst, *dst, *c);
            }
        }
        if let Some(store) = &plan.store {
            self.spill_store(code, store.src, store.offset);
        }
        Ok(())
    }

    fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
        x86_64::emit_movaps(code, dst, src);
    }

    fn emit_store(
        &mut self,
        code: &mut Vec<u8>,
        src: Reg,
        offset: u32,
    ) -> Result<(), &'static str> {
        self.spill_store(code, src, offset);
        Ok(())
    }

    fn emit_resolve(
        &mut self,
        code: &mut Vec<u8>,
        vid: regalloc::ValueId,
        target: Reg,
        reg_for: &[Option<Reg>],
        spill_for: &[Option<u32>],
        remat_for: &[Option<u32>],
    ) -> Reg {
        let idx = vid.0 as usize;
        if let Some(Some(reg)) = reg_for.get(idx) {
            *reg
        } else if let Some(Some(bits)) = remat_for.get(idx) {
            x86_64::emit_const(code, target, f32::from_bits(*bits), X86_BUILTIN_SCRATCH);
            target
        } else if let Some(Some(offset)) = spill_for.get(idx) {
            self.spill_load(code, target, *offset);
            target
        } else {
            panic!(
                "value {:?} has no register, spill slot, or rematerialize entry",
                vid
            );
        }
    }

    fn emit_skip_if_all_false(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> usize {
        x86_64::emit_movmskps_eax(code, mask_reg);
        x86_64::emit_test_eax(code);
        x86_64::emit_jcc_rel32(code, 0x84) // jz: taken when eax == 0 (all lanes false)
    }

    fn emit_skip_if_all_true(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> usize {
        x86_64::emit_movmskps_eax(code, mask_reg);
        x86_64::emit_cmp_eax_imm8(code, 0x0F);
        x86_64::emit_jcc_rel32(code, 0x84) // je: taken when eax == 0xF (all lanes true)
    }

    fn emit_jump(&mut self, code: &mut Vec<u8>) -> usize {
        x86_64::emit_jmp_rel32(code)
    }

    fn patch_branch(&mut self, code: &mut Vec<u8>, branch: usize, target: usize) {
        x86_64::patch_rel32(code, branch, target);
    }

    fn epilogue(&mut self, code: &mut Vec<u8>, result_reg: Reg, _frame_size: u32) {
        if result_reg.0 != 0 {
            x86_64::emit_movaps(code, Reg(0), result_reg);
        }
        if self.frame_bytes > 0 {
            x86_64::emit_add_rsp(code, self.frame_bytes);
        }
        code.push(0xC3); // RET
    }

    fn select_extra_reload(&self) -> Reg {
        // xmm13: builtin scratch, unused by `emit_select` (whose internal
        // temp is X86_SCRATCH/xmm10) and by the reload path (movups /
        // RIP-relative consts touch no scratch).
        Reg(13)
    }

    // SysV: rdi = ctx (read-only in the body's gathers), rsi = out,
    // rdx = groups, xmm0..3 = x0/y/z/w. Loop registers: r8 = counter; the
    // body's scratch GPRs are rax/rcx (gather, movmskps) — disjoint. In
    // red-zone mode (`frame_bytes == 0`) the body spills below rsp, so the
    // coordinate slots allocated here at [rsp..) never collide with it.
    fn emit_collapse_loop(
        &mut self,
        hoist: &[u8],
        body: &[u8],
        result_reg: Reg,
        _frame_size: u32,
        hoist_slots: u32,
    ) -> Result<Vec<u8>, &'static str> {
        const VW: u32 = 16;
        let f = self.frame_bytes;
        let total = f + (4 + hoist_slots) * VW;
        let slot = |k: u32| (f + k * VW) as i32;
        let mut code: Vec<u8> = Vec::with_capacity(hoist.len() + body.len() + 128);

        x86_64::emit_sub_rsp(&mut code, total);
        for k in 0..4u32 {
            x86_64::emit_movups_store_rsp32(&mut code, Reg(k as u8), slot(k));
        }
        // LICM prologue: X-invariant values, parked in the hoist slots.
        code.extend_from_slice(hoist);
        // xor r8, r8 — batch counter.
        code.extend_from_slice(&[0x4D, 0x31, 0xC0]);

        let loop_top = code.len();
        // cmp r8, rdx ; jae end (forward, patched below).
        code.extend_from_slice(&[0x49, 0x39, 0xD0]);
        code.extend_from_slice(&[0x0F, 0x83]);
        let jae_at = code.len();
        code.extend_from_slice(&[0, 0, 0, 0]);

        for k in 0..4u32 {
            x86_64::emit_movups_load_rsp32(&mut code, Reg(k as u8), slot(k));
        }
        code.extend_from_slice(body);

        // out[i] = result ; add rsi, 16.
        x86_64::emit_movups_store_base(&mut code, result_reg, 6);
        code.extend_from_slice(&[0x48, 0x83, 0xC6, VW as u8]);

        // X += lane count (xmm0/1 are about to be reloaded next iteration).
        x86_64::emit_movups_load_rsp32(&mut code, Reg(0), slot(0));
        x86_64::emit_const(&mut code, Reg(1), (VW / 4) as f32, X86_BUILTIN_SCRATCH);
        x86_64::emit_binary(&mut code, OpKind::Add, Reg(0), Reg(0), Reg(1));
        x86_64::emit_movups_store_rsp32(&mut code, Reg(0), slot(0));

        // inc r8 ; jmp loop_top (backward).
        code.extend_from_slice(&[0x49, 0xFF, 0xC0]);
        code.push(0xE9);
        let jmp_at = code.len();
        code.extend_from_slice(&[0, 0, 0, 0]);

        let end = code.len();
        x86_64::emit_add_rsp(&mut code, total);
        code.push(0xC3); // RET

        let jae_rel = (end as i32) - (jae_at as i32 + 4);
        code[jae_at..jae_at + 4].copy_from_slice(&jae_rel.to_le_bytes());
        let jmp_rel = (loop_top as i32) - (jmp_at as i32 + 4);
        code[jmp_at..jmp_at + 4].copy_from_slice(&jmp_rel.to_le_bytes());
        Ok(code)
    }
}

// =============================================================================
// AVX-512 backend: the leaf emit for the SHARED driver, 512-bit (zmm) kernels.
// =============================================================================
//
// Mirrors `X86Backend`'s register roles exactly so it reuses the shared driver,
// `linear_scan` budget, and the RELOAD_REGS/SCRATCH_BASE consts unchanged — only
// the leaf encodings differ. EVEX is 3-operand and non-destructive, so there is
// no SSE two-operand hazard (operands never clobbered; may alias dst), and we
// have real hardware FMA.
//
// Register roles (zmm): zmm0-3 inputs, zmm4-9 allocatable (6), zmm10 fixed
// scratch (FMA temp), zmm11-12 reload (RELOAD_REGS), zmm13-15 builtin scratch.
//
// Spills use a REAL stack frame, not the red zone: a zmm slot is 64 bytes, so
// even one spill overflows the 128-byte red zone. `FrameLayout` hands out
// offsets in 16-byte units (the shared layout assumes 128-bit slots); scale ×4
// to 64-byte zmm slots, and allocate `frame_size * 4` in the prologue.
//
// Scope (Stage 1): the arithmetic subset (Var/Const/Unary{sqrt,neg,abs}/Binary
// {add,sub,mul,div,min,max}/FMA). Select (k-mask class), Clamp, and the
// transcendentals reject loudly and are later stages.

/// Scale a `FrameLayout` 16-byte slot offset to a 64-byte zmm frame offset.
#[cfg(target_arch = "x86_64")]
fn avx512_slot_disp(offset: u32) -> i32 {
    // FrameLayout packs slots at 16-byte stride; zmm needs 64. Slots live at
    // [rsp + 0], [rsp + 64], ... within the frame we allocate in the prologue.
    (offset as i32 / 16) * 64
}

/// Total zmm frame bytes for a `FrameLayout.frame_size` (16-byte units).
#[cfg(target_arch = "x86_64")]
fn avx512_frame_bytes(frame_size: u32) -> u32 {
    (frame_size / 16) * 64
}

/// AVX-512 implementation of the shared driver's leaf operations.
#[cfg(target_arch = "x86_64")]
struct Avx512Backend;

#[cfg(target_arch = "x86_64")]
impl Avx512Backend {
    fn reload(code: &mut Vec<u8>, reload: &Reload) {
        match reload {
            Reload::FromStack { target, offset } => {
                avx512::emit_load_rsp(code, *target, avx512_slot_disp(*offset));
            }
            Reload::Const { target, val_bits } => {
                avx512::emit_const(code, *target, f32::from_bits(*val_bits));
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl IsaBackend for Avx512Backend {
    type Branch = usize;

    fn num_regs(&self) -> u8 {
        X86_SCHED_NUM_REGS // same 6 allocatable (zmm4-9)
    }

    fn frame_bytes(&self, frame_size: u32) -> u32 {
        avx512_frame_bytes(frame_size)
    }

    fn begin(
        &mut self,
        _schedule: &[(regalloc::ValueId, ScheduledOp)],
    ) -> Result<(), &'static str> {
        Ok(()) // const broadcast is self-contained; no pool.
    }

    fn prologue(&mut self, code: &mut Vec<u8>, frame_size: u32) {
        let bytes = avx512_frame_bytes(frame_size);
        if bytes > 0 {
            avx512::emit_sub_rsp(code, bytes);
        }
    }

    fn emit_plan(
        &mut self,
        code: &mut Vec<u8>,
        plan: &InstructionPlan,
    ) -> Result<(), &'static str> {
        for r in &plan.reloads {
            Self::reload(code, r);
        }
        if let Some((dst, src)) = plan.setup_mov {
            avx512::emit_mov(code, dst, src);
        }
        match &plan.op {
            ResolvedOp::Nop => {}
            ResolvedOp::LoadConst { dst, val_bits } => {
                avx512::emit_const(code, *dst, f32::from_bits(*val_bits));
            }
            ResolvedOp::Unary { op, dst, src } => {
                avx512::emit_unary(code, *op, *dst, *src)?;
            }
            ResolvedOp::ShiftImm {
                op,
                dst,
                src,
                amount,
            } => {
                avx512::emit_shift_imm(code, *op, *dst, *src, *amount)?;
            }
            ResolvedOp::Gather { dst, idx, slot } => {
                // dst = buffer[slot][idx]. The context pointer (array of buffer
                // base pointers) is caller-provided in rdi; arithmetic/const emit
                // never touches rdi, so it survives to here. zmm13/zmm14 are the
                // backend's reserved non-allocatable scratch (see UNARY_SCRATCH).
                const IDX_INT: Reg = Reg(13);
                const GATHER_DST: Reg = Reg(14);
                const RAX: u8 = 0;
                const RDI: u8 = 7;
                avx512::emit_cvttps2dq(code, IDX_INT, *idx); // float idx -> int32 lanes
                avx512::emit_set_gather_mask(code); // k1 = 0xFFFF (clobbers rax)
                avx512::emit_load_ptr_from_ctx(code, RAX, RDI, (*slot as i32) * 8);
                avx512::emit_gather(code, GATHER_DST, RAX, IDX_INT);
                avx512::emit_mov(code, *dst, GATHER_DST);
            }
            ResolvedOp::Binary {
                op,
                dst,
                left,
                right,
            } => {
                // EVEX 3-operand: no two-operand hazard, emit directly.
                // Comparisons produce a vector mask (vcmpps -> vpmovm2d).
                if avx512::is_compare(*op) {
                    avx512::emit_compare(code, *op, *dst, *left, *right)?;
                } else {
                    avx512::emit_binary(code, *op, *dst, *left, *right)?;
                }
            }
            ResolvedOp::FusedMulAdd { dst, a, b } => {
                // dst holds c (setup_mov); real FMA231: dst = a*b + dst.
                avx512::emit_fmadd_c_in_dst(code, *dst, *a, *b);
            }
            ResolvedOp::DecomposedMulAdd {
                dst,
                a,
                b,
                c,
                c_deferred,
            } => {
                // dst = a*b, reload c (after the multiply if deferred), dst += c.
                avx512::emit_binary(code, OpKind::Mul, *dst, *a, *b)?;
                match c_deferred {
                    Some(DeferredReload::FromStack(off)) => {
                        avx512::emit_load_rsp(code, *c, avx512_slot_disp(*off));
                    }
                    Some(DeferredReload::Const(bits)) => {
                        avx512::emit_const(code, *c, f32::from_bits(*bits));
                    }
                    None => {}
                }
                avx512::emit_binary(code, OpKind::Add, *dst, *dst, *c)?;
            }
            ResolvedOp::Select {
                dst,
                if_true,
                if_false,
            } => {
                // setup_mov already placed the vector mask in dst; one vpternlogd.
                avx512::emit_select(code, *dst, *if_true, *if_false);
            }
        }
        if let Some(store) = &plan.store {
            avx512::emit_store_rsp(code, store.src, avx512_slot_disp(store.offset));
        }
        Ok(())
    }

    fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
        avx512::emit_mov(code, dst, src);
    }

    fn emit_store(
        &mut self,
        code: &mut Vec<u8>,
        src: Reg,
        offset: u32,
    ) -> Result<(), &'static str> {
        avx512::emit_store_rsp(code, src, avx512_slot_disp(offset));
        Ok(())
    }

    fn emit_resolve(
        &mut self,
        code: &mut Vec<u8>,
        vid: regalloc::ValueId,
        target: Reg,
        reg_for: &[Option<Reg>],
        spill_for: &[Option<u32>],
        remat_for: &[Option<u32>],
    ) -> Reg {
        let idx = vid.0 as usize;
        if let Some(Some(reg)) = reg_for.get(idx) {
            *reg
        } else if let Some(Some(bits)) = remat_for.get(idx) {
            avx512::emit_const(code, target, f32::from_bits(*bits));
            target
        } else if let Some(Some(offset)) = spill_for.get(idx) {
            avx512::emit_load_rsp(code, target, avx512_slot_disp(*offset));
            target
        } else {
            panic!(
                "value {:?} has no register, spill slot, or rematerialize entry",
                vid
            );
        }
    }

    // Select short-circuit guards: reduce the vector mask to flags (vptestmd +
    // kortestw) and branch. jz = all-false (skip true arm); jc = all-true (skip
    // false arm). Mirrors the SSE2 MOVMSKPS guards, k-register-based.
    fn emit_skip_if_all_false(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> usize {
        avx512::emit_mask_flags(code, mask_reg);
        x86_64::emit_jcc_rel32(code, 0x84) // jz: ZF set when k1 == 0 (all false)
    }
    fn emit_skip_if_all_true(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> usize {
        avx512::emit_mask_flags(code, mask_reg);
        x86_64::emit_jcc_rel32(code, 0x82) // jc: CF set when k1 == 0xFFFF (all true)
    }
    fn emit_jump(&mut self, code: &mut Vec<u8>) -> usize {
        x86_64::emit_jmp_rel32(code)
    }
    fn patch_branch(&mut self, code: &mut Vec<u8>, branch: usize, target: usize) {
        x86_64::patch_rel32(code, branch, target);
    }

    fn epilogue(&mut self, code: &mut Vec<u8>, result_reg: Reg, frame_size: u32) {
        if result_reg.0 != 0 {
            avx512::emit_mov(code, Reg(0), result_reg);
        }
        let bytes = avx512_frame_bytes(frame_size);
        if bytes > 0 {
            avx512::emit_add_rsp(code, bytes);
        }
        avx512::emit_ret(code);
    }

    fn select_extra_reload(&self) -> Reg {
        // zmm13: outside the allocatable range and the reload pair;
        // `vpternlogd` consumes its three operands with no temp.
        Reg(13)
    }

    // Same register roles as `X86Backend::emit_collapse_loop` at zmm width;
    // AVX-512 always uses an allocated frame (no red-zone mode), so the
    // body's slots sit at [rsp..bytes) and the coordinate slots above them.
    fn emit_collapse_loop(
        &mut self,
        hoist: &[u8],
        body: &[u8],
        result_reg: Reg,
        frame_size: u32,
        hoist_slots: u32,
    ) -> Result<Vec<u8>, &'static str> {
        const VW: u32 = 64;
        let f = avx512_frame_bytes(frame_size);
        let total = f + (4 + hoist_slots) * VW;
        let slot = |k: u32| (f + k * VW) as i32;
        let mut code: Vec<u8> = Vec::with_capacity(hoist.len() + body.len() + 128);

        avx512::emit_sub_rsp(&mut code, total);
        for k in 0..4u32 {
            avx512::emit_store_rsp(&mut code, Reg(k as u8), slot(k));
        }
        // LICM prologue: X-invariant values, parked in the hoist slots.
        code.extend_from_slice(hoist);
        // xor r8, r8 — batch counter.
        code.extend_from_slice(&[0x4D, 0x31, 0xC0]);

        let loop_top = code.len();
        // cmp r8, rdx ; jae end (forward, patched below).
        code.extend_from_slice(&[0x49, 0x39, 0xD0]);
        code.extend_from_slice(&[0x0F, 0x83]);
        let jae_at = code.len();
        code.extend_from_slice(&[0, 0, 0, 0]);

        for k in 0..4u32 {
            avx512::emit_load_rsp(&mut code, Reg(k as u8), slot(k));
        }
        code.extend_from_slice(body);

        // out[i] = result ; add rsi, 64.
        avx512::emit_store_zmm_base(&mut code, result_reg, 6);
        code.extend_from_slice(&[0x48, 0x83, 0xC6, VW as u8]);

        // X += lane count.
        avx512::emit_load_rsp(&mut code, Reg(0), slot(0));
        avx512::emit_const(&mut code, Reg(1), (VW / 4) as f32);
        avx512::emit_binary(&mut code, OpKind::Add, Reg(0), Reg(0), Reg(1))?;
        avx512::emit_store_rsp(&mut code, Reg(0), slot(0));

        // inc r8 ; jmp loop_top (backward).
        code.extend_from_slice(&[0x49, 0xFF, 0xC0]);
        code.push(0xE9);
        let jmp_at = code.len();
        code.extend_from_slice(&[0, 0, 0, 0]);

        let end = code.len();
        avx512::emit_add_rsp(&mut code, total);
        avx512::emit_ret(&mut code);

        let jae_rel = (end as i32) - (jae_at as i32 + 4);
        code[jae_at..jae_at + 4].copy_from_slice(&jae_rel.to_le_bytes());
        let jmp_rel = (loop_top as i32) - (jmp_at as i32 + 4);
        code[jmp_at..jmp_at + 4].copy_from_slice(&jmp_rel.to_le_bytes());
        Ok(code)
    }
}

// =============================================================================
// AVX2 backend: the leaf emit for the SHARED driver, 256-bit (ymm) kernels.
// =============================================================================
//
// The middle width between `X86Backend` (128-bit) and `Avx512Backend`
// (512-bit). Register roles are IDENTICAL to `X86Backend` — same input/
// allocatable/scratch/reload layout (ymm0-15 is the same physical file as
// xmm0-15) — only the leaf encodings differ (VEX.256 instead of legacy SSE).
// VEX is 3-operand like EVEX, so — like `Avx512Backend` — there is no
// two-operand hazard to route around.
//
// Spills use a real stack frame (mirrors `Avx512Backend`'s reasoning): a ymm
// slot is 32 bytes, so `FrameLayout`'s 16-byte-unit offsets are scaled ×2.

/// Scale a `FrameLayout` 16-byte slot offset to a 32-byte ymm frame offset.
#[cfg(target_arch = "x86_64")]
fn avx2_slot_disp(offset: u32) -> i32 {
    (offset as i32 / 16) * 32
}

/// Total ymm frame bytes for a `FrameLayout.frame_size` (16-byte units).
#[cfg(target_arch = "x86_64")]
fn avx2_frame_bytes(frame_size: u32) -> u32 {
    (frame_size / 16) * 32
}

/// AVX2 implementation of the shared driver's leaf operations.
#[cfg(target_arch = "x86_64")]
struct Avx2Backend;

#[cfg(target_arch = "x86_64")]
impl Avx2Backend {
    fn reload(code: &mut Vec<u8>, reload: &Reload) {
        match reload {
            Reload::FromStack { target, offset } => {
                avx2::emit_load_rsp(code, *target, avx2_slot_disp(*offset));
            }
            Reload::Const { target, val_bits } => {
                avx2::emit_const(code, *target, f32::from_bits(*val_bits));
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl IsaBackend for Avx2Backend {
    type Branch = usize;

    fn num_regs(&self) -> u8 {
        AVX2_SCHED_NUM_REGS
    }

    fn frame_bytes(&self, frame_size: u32) -> u32 {
        avx2_frame_bytes(frame_size)
    }

    fn begin(
        &mut self,
        _schedule: &[(regalloc::ValueId, ScheduledOp)],
    ) -> Result<(), &'static str> {
        Ok(()) // const broadcast is self-contained; no pool.
    }

    fn prologue(&mut self, code: &mut Vec<u8>, frame_size: u32) {
        let bytes = avx2_frame_bytes(frame_size);
        if bytes > 0 {
            avx2::emit_sub_rsp(code, bytes);
        }
    }

    fn emit_plan(
        &mut self,
        code: &mut Vec<u8>,
        plan: &InstructionPlan,
    ) -> Result<(), &'static str> {
        for r in &plan.reloads {
            Self::reload(code, r);
        }
        if let Some((dst, src)) = plan.setup_mov {
            avx2::emit_mov(code, dst, src);
        }
        match &plan.op {
            ResolvedOp::Nop => {}
            ResolvedOp::LoadConst { dst, val_bits } => {
                avx2::emit_const(code, *dst, f32::from_bits(*val_bits));
            }
            ResolvedOp::Unary { op, dst, src } => {
                avx2::emit_unary(code, *op, *dst, *src)?;
            }
            ResolvedOp::ShiftImm {
                op,
                dst,
                src,
                amount,
            } => {
                avx2::emit_shift_imm(code, *op, *dst, *src, *amount)?;
            }
            ResolvedOp::Gather { dst, idx, slot } => {
                // Context pointer (array of buffer base pointers) arrives in
                // rdi; arithmetic/const emit never touches rdi, so it
                // survives to here. ymm13/14 mirror X86Backend's gather
                // scratch; ymm8/9 are the AVX2-only high-half scratch this
                // two-half gather needs (see `avx2::emit_gather_scalar`).
                // ymm8/9 are non-allocatable by construction — see
                // `AVX2_SCHED_NUM_REGS`, which caps the pool at ymm4-7 so the
                // allocator can never place `dst`/`idx` where this clobbers.
                avx2::emit_gather_scalar(
                    code,
                    *dst,
                    *idx,
                    *slot,
                    x86_64::GatherScratch {
                        base_gpr: 0,  // rax
                        index_gpr: 1, // rcx
                        ctx_gpr: 7,   // rdi
                        idx_lanes: Reg(13),
                        value: Reg(14),
                    },
                    Reg(9),
                    Reg(8),
                );
            }
            ResolvedOp::Binary {
                op,
                dst,
                left,
                right,
            } => {
                // VEX 3-operand: no two-operand hazard, emit directly.
                avx2::emit_binary(code, *op, *dst, *left, *right)?;
            }
            ResolvedOp::FusedMulAdd { dst, a, b } => {
                avx2::emit_fmadd_c_in_dst(code, *dst, *a, *b);
            }
            ResolvedOp::DecomposedMulAdd {
                dst,
                a,
                b,
                c,
                c_deferred,
            } => {
                avx2::emit_binary(code, OpKind::Mul, *dst, *a, *b)?;
                match c_deferred {
                    Some(DeferredReload::FromStack(off)) => {
                        avx2::emit_load_rsp(code, *c, avx2_slot_disp(*off));
                    }
                    Some(DeferredReload::Const(bits)) => {
                        avx2::emit_const(code, *c, f32::from_bits(*bits));
                    }
                    None => {}
                }
                avx2::emit_binary(code, OpKind::Add, *dst, *dst, *c)?;
            }
            ResolvedOp::Select {
                dst,
                if_true,
                if_false,
            } => {
                // setup_mov already placed the vector mask in dst.
                avx2::emit_select(code, *dst, *if_true, *if_false);
            }
        }
        if let Some(store) = &plan.store {
            avx2::emit_store_rsp(code, store.src, avx2_slot_disp(store.offset));
        }
        Ok(())
    }

    fn emit_mov(&mut self, code: &mut Vec<u8>, dst: Reg, src: Reg) {
        avx2::emit_mov(code, dst, src);
    }

    fn emit_store(
        &mut self,
        code: &mut Vec<u8>,
        src: Reg,
        offset: u32,
    ) -> Result<(), &'static str> {
        avx2::emit_store_rsp(code, src, avx2_slot_disp(offset));
        Ok(())
    }

    fn emit_resolve(
        &mut self,
        code: &mut Vec<u8>,
        vid: regalloc::ValueId,
        target: Reg,
        reg_for: &[Option<Reg>],
        spill_for: &[Option<u32>],
        remat_for: &[Option<u32>],
    ) -> Reg {
        let idx = vid.0 as usize;
        if let Some(Some(reg)) = reg_for.get(idx) {
            *reg
        } else if let Some(Some(bits)) = remat_for.get(idx) {
            avx2::emit_const(code, target, f32::from_bits(*bits));
            target
        } else if let Some(Some(offset)) = spill_for.get(idx) {
            avx2::emit_load_rsp(code, target, avx2_slot_disp(*offset));
            target
        } else {
            panic!(
                "value {:?} has no register, spill slot, or rematerialize entry",
                vid
            );
        }
    }

    // Select short-circuit guards: vmovmskps -> eax[7:0], same shape as
    // X86Backend's MOVMSKPS guards but 8 lanes wide (al == 0xFF for
    // all-true, not 0x0F — see `avx2::emit_cmp_al_imm8`'s doc for why the
    // sign-extending `cmp eax, imm8` X86Backend uses doesn't work here).
    fn emit_skip_if_all_false(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> usize {
        avx2::emit_movmskps_eax(code, mask_reg);
        x86_64::emit_test_eax(code);
        x86_64::emit_jcc_rel32(code, 0x84) // jz: taken when eax == 0 (all lanes false)
    }

    fn emit_skip_if_all_true(&mut self, code: &mut Vec<u8>, mask_reg: Reg) -> usize {
        avx2::emit_movmskps_eax(code, mask_reg);
        avx2::emit_cmp_al_imm8(code, 0xFF);
        x86_64::emit_jcc_rel32(code, 0x84) // je: taken when al == 0xFF (all lanes true)
    }

    fn emit_jump(&mut self, code: &mut Vec<u8>) -> usize {
        x86_64::emit_jmp_rel32(code)
    }
    fn patch_branch(&mut self, code: &mut Vec<u8>, branch: usize, target: usize) {
        x86_64::patch_rel32(code, branch, target);
    }

    fn epilogue(&mut self, code: &mut Vec<u8>, result_reg: Reg, frame_size: u32) {
        if result_reg.0 != 0 {
            avx2::emit_mov(code, Reg(0), result_reg);
        }
        let bytes = avx2_frame_bytes(frame_size);
        if bytes > 0 {
            avx2::emit_add_rsp(code, bytes);
        }
        avx2::emit_ret(code);
    }

    fn select_extra_reload(&self) -> Reg {
        // ymm13: outside the allocatable range and the reload pair; the
        // AVX2 select is a VEX blend with no internal temp.
        Reg(13)
    }

    // Same register roles as `X86Backend::emit_collapse_loop` at ymm width;
    // AVX2 always uses an allocated frame (mirrors AVX-512).
    fn emit_collapse_loop(
        &mut self,
        hoist: &[u8],
        body: &[u8],
        result_reg: Reg,
        frame_size: u32,
        hoist_slots: u32,
    ) -> Result<Vec<u8>, &'static str> {
        const VW: u32 = 32;
        let f = avx2_frame_bytes(frame_size);
        let total = f + (4 + hoist_slots) * VW;
        let slot = |k: u32| (f + k * VW) as i32;
        let mut code: Vec<u8> = Vec::with_capacity(hoist.len() + body.len() + 128);

        avx2::emit_sub_rsp(&mut code, total);
        for k in 0..4u32 {
            avx2::emit_store_rsp(&mut code, Reg(k as u8), slot(k));
        }
        // LICM prologue: X-invariant values, parked in the hoist slots.
        code.extend_from_slice(hoist);
        // xor r8, r8 — batch counter.
        code.extend_from_slice(&[0x4D, 0x31, 0xC0]);

        let loop_top = code.len();
        // cmp r8, rdx ; jae end (forward, patched below).
        code.extend_from_slice(&[0x49, 0x39, 0xD0]);
        code.extend_from_slice(&[0x0F, 0x83]);
        let jae_at = code.len();
        code.extend_from_slice(&[0, 0, 0, 0]);

        for k in 0..4u32 {
            avx2::emit_load_rsp(&mut code, Reg(k as u8), slot(k));
        }
        code.extend_from_slice(body);

        // out[i] = result ; add rsi, 32.
        avx2::emit_store_base(&mut code, result_reg, 6);
        code.extend_from_slice(&[0x48, 0x83, 0xC6, VW as u8]);

        // X += lane count.
        avx2::emit_load_rsp(&mut code, Reg(0), slot(0));
        avx2::emit_const(&mut code, Reg(1), (VW / 4) as f32);
        avx2::emit_binary(&mut code, OpKind::Add, Reg(0), Reg(0), Reg(1))?;
        avx2::emit_store_rsp(&mut code, Reg(0), slot(0));

        // inc r8 ; jmp loop_top (backward).
        code.extend_from_slice(&[0x49, 0xFF, 0xC0]);
        code.push(0xE9);
        let jmp_at = code.len();
        code.extend_from_slice(&[0, 0, 0, 0]);

        let end = code.len();
        avx2::emit_add_rsp(&mut code, total);
        avx2::emit_ret(&mut code);

        let jae_rel = (end as i32) - (jae_at as i32 + 4);
        code[jae_at..jae_at + 4].copy_from_slice(&jae_rel.to_le_bytes());
        let jmp_rel = (loop_top as i32) - (jmp_at as i32 + 4);
        code[jmp_at..jmp_at + 4].copy_from_slice(&jmp_rel.to_le_bytes());
        Ok(code)
    }
}

/// Compile an arena DAG to an AVX2 (256-bit, 8-lane ymm) kernel via the shared
/// driver. Same arg shape as [`compile_arena_dag`] but the kernel's ABI is
/// `__m256` (one pixel per lane, 8 pixels per call).
#[cfg(target_arch = "x86_64")]
pub fn compile_arena_dag_avx2(
    arena: &ExprArena,
    root: ExprId,
) -> Result<CompileResult, &'static str> {
    let schedule = arena_to_schedule(arena, root);
    let uses = arena_to_uses(&schedule);
    compile_dag_via_backend(schedule, uses, &mut Avx2Backend)
}

/// Compile an arena DAG to an AVX-512 (512-bit, 16-lane zmm) kernel via the
/// shared driver. Same arg shape as [`compile_arena_dag`] but the kernel's ABI
/// is `__m512` (one pixel per lane, 16 pixels per call). Stage-1 arithmetic
/// subset only; ops outside it return `Err`.
#[cfg(target_arch = "x86_64")]
pub fn compile_arena_dag_avx512(
    arena: &ExprArena,
    root: ExprId,
) -> Result<CompileResult, &'static str> {
    let schedule = arena_to_schedule(arena, root);
    let uses = arena_to_uses(&schedule);
    compile_dag_via_backend(schedule, uses, &mut Avx512Backend)
}

/// Compile an [`ExprArena`] DAG into a **collapse** kernel: the row loop is
/// emitted *inside* the code, so one call fills `groups` output batches with
/// no per-batch Rust↔JIT boundary. This is the internal-loop realization of a
/// lattice collapse — the point of the whole design.
///
/// The per-batch body (produced by [`emit_dag_body`], with derivatives /
/// reductions / gathers / transcendentals already lowered — the same Stage-1
/// chain as [`compile_arena_dag`]) is wrapped in the build width's
/// [`IsaBackend::emit_collapse_loop`] scaffold: X is an induction value
/// stepped by the batch width each iteration, Y/Z/W are loop-invariant
/// arguments, gathers read buffer bases from the context register, and each
/// batch stores straight to `out`. Matches the
/// [`CollapseKernelFn`](executable::CollapseKernelFn) ABI
/// `(ctx, out, groups, x0, y, z, w)`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub fn compile_collapse(
    arena: &crate::arena::ExprArena,
    root: crate::arena::ExprId,
) -> Result<CompileResult, &'static str> {
    let (arena, root) = lowering::lower_dwrt_owned(arena, root)?;
    let (arena, root) = lowering::expand_reduce_owned(&arena, root);
    let (arena, root) = lowering::expand_gather_owned(&arena, root);
    let (arena, root) = lowering::expand_transcendentals_owned(&arena, root);
    let arena = &arena;
    let schedule = arena_to_schedule(arena, root);
    let uses = arena_to_uses(&schedule);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        compile_collapse_via_backend(schedule, uses, &mut Avx512Backend)
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
    {
        compile_collapse_via_backend(schedule, uses, &mut Avx2Backend)
    }
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    {
        compile_collapse_via_backend(schedule, uses, &mut X86Backend::default())
    }
    #[cfg(target_arch = "aarch64")]
    {
        let mut backend = Aarch64Backend {
            pool: ConstPool::new(),
            adr_patch_pos: 0,
            max_regs: EmitCtx::default().max_regs,
        };
        compile_collapse_via_backend(schedule, uses, &mut backend)
    }
}

/// Drive a `(schedule, uses_map)` to a complete collapse kernel via an
/// [`IsaBackend`]: the body from [`emit_dag_body`], framed by the backend's
/// [`IsaBackend::emit_collapse_loop`] scaffold.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn compile_collapse_via_backend<B: IsaBackend>(
    schedule: Vec<(regalloc::ValueId, ScheduledOp)>,
    uses_map: Vec<Vec<regalloc::ValueId>>,
    backend: &mut B,
) -> Result<CompileResult, &'static str> {
    let variance = schedule_variance(&schedule);
    let Some(plan) = plan_collapse_hoist(&schedule, &variance) else {
        // Nothing X-invariant worth hoisting: the plain loop.
        let (body, result_reg, frame_size, spill_count) =
            emit_dag_body(schedule, uses_map, backend)?;
        let code = backend.emit_collapse_loop(&[], &body, result_reg, frame_size, 0)?;
        let exec = unsafe { executable::ExecutableCode::from_code(&code)? };
        return Ok(CompileResult {
            code: exec,
            spill_count,
            spill_bytes: backend.frame_bytes(frame_size),
            max_regs: backend.num_regs(),
            hoisted_values: 0,
        });
    };

    // The prologue and the loop body share one stack frame: spill slots in
    // [0, m), the scaffold's four coordinate slots at [m, m+64), hoist slots
    // above those. `m` is the max of the two frames (each region is only live
    // while its code runs; the hoist slots outlive both). Allocation is pure,
    // so pre-sizing here computes exactly the frames the emissions below will.
    // The 144 floor keeps x86's SSE2 backend out of red-zone mode — hoist
    // offsets are far past the zone, so both emissions must latch
    // allocated-frame ([rsp + offset]) addressing.
    let pf = presize_frame(&plan.prologue, backend.num_regs())?;
    let bf = presize_frame(&plan.body, backend.num_regs())?;
    let m = pf.max(bf).max(144);
    let hoist_map: alloc::collections::BTreeMap<regalloc::ValueId, u32> = plan
        .roots
        .iter()
        .enumerate()
        .map(|(i, vid)| (*vid, m + 4 * 16 + (i as u32) * 16))
        .collect();

    let uses_p = arena_to_uses(&plan.prologue);
    let (prologue, _, _, pro_spills) = emit_dag_body_hoisted(
        plan.prologue,
        uses_p,
        backend,
        HoistCtx::Prologue(&hoist_map),
        Some(m),
    )?;
    let uses_b = arena_to_uses(&plan.body);
    let (body, result_reg, _, body_spills) = emit_dag_body_hoisted(
        plan.body,
        uses_b,
        backend,
        HoistCtx::Body(&hoist_map),
        Some(m),
    )?;

    let code =
        backend.emit_collapse_loop(&prologue, &body, result_reg, m, plan.roots.len() as u32)?;
    let exec = unsafe { executable::ExecutableCode::from_code(&code)? };
    Ok(CompileResult {
        code: exec,
        spill_count: pro_spills + body_spills,
        spill_bytes: backend.frame_bytes(m),
        max_regs: backend.num_regs(),
        hoisted_values: plan.roots.len() as u32,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_arch = "aarch64")]
    use crate::arena::ExprArena;

    /// A `Dwrt` that reaches the scheduler (a caller bypassed the lowering
    /// pipeline) must fail loudly at the schedule boundary, not as a cryptic
    /// emit panic. The compile entry points run `lower_dwrt` first, so this
    /// exercises calling `arena_to_schedule` directly.
    #[test]
    #[should_panic(expected = "Dwrt (autodiff) node reached the JIT")]
    fn surviving_dwrt_fails_loudly() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let v = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, x, v);
        let _ = arena_to_schedule(&a, root);
    }

    // =========================================================================
    // Graph Coloring (DAG) Tests
    // =========================================================================

    // =========================================================================
    // FrameLayout unit tests
    // =========================================================================

    #[test]
    fn frame_layout_empty() {
        let layout = FrameLayout::from_allocation(&[]).unwrap();
        assert_eq!(layout.frame_size, 0);
        assert!(layout.spill_slots.is_empty());
    }

    #[test]
    fn frame_layout_one_spill() {
        let layout = FrameLayout::from_allocation(&[regalloc::ValueId(5)]).unwrap();
        assert_eq!(layout.frame_size, 16);
        assert_eq!(layout.spill_slots[&regalloc::ValueId(5)], 0);
    }

    #[test]
    fn frame_layout_alignment() {
        let spilled = [
            regalloc::ValueId(1),
            regalloc::ValueId(2),
            regalloc::ValueId(3),
        ];
        let layout = FrameLayout::from_allocation(&spilled).unwrap();
        // 3 * 16 = 48, already aligned
        assert_eq!(layout.frame_size, 48);
        assert_eq!(layout.spill_slots[&regalloc::ValueId(1)], 0);
        assert_eq!(layout.spill_slots[&regalloc::ValueId(2)], 16);
        assert_eq!(layout.spill_slots[&regalloc::ValueId(3)], 32);
    }

    // =========================================================================
    // resolve_operands unit tests — the spill logic that was buggy
    // =========================================================================

    /// Helper: build minimal assignment + spill maps for resolve_operands tests.
    fn make_maps(
        assigned: &[(u32, u8)],
        spilled: &[(u32, u32)],
    ) -> (
        alloc::collections::BTreeMap<regalloc::ValueId, Reg>,
        alloc::collections::BTreeMap<regalloc::ValueId, u32>,
        alloc::collections::BTreeMap<regalloc::ValueId, u32>,
    ) {
        use alloc::collections::BTreeMap;
        let mut assignment = BTreeMap::new();
        for &(v, r) in assigned {
            assignment.insert(regalloc::ValueId(v), Reg(r));
        }
        let mut slots = BTreeMap::new();
        for &(v, off) in spilled {
            slots.insert(regalloc::ValueId(v), off);
        }
        (assignment, slots, BTreeMap::new())
    }

    #[test]
    fn resolve_binary_no_spills() {
        // left=v4, right=v5, dst=v6 — all in registers
        let (assign, spills, remat) = make_maps(&[(0, 4), (1, 5), (2, 6)], &[]);
        let op = ScheduledOp::Binary(OpKind::Add, regalloc::ValueId(0), regalloc::ValueId(1));
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(6)), &assign, &spills, &remat, Reg(13)).unwrap();

        assert!(plan.reloads.is_empty());
        assert!(plan.store.is_none());
        assert_eq!(
            plan.op,
            ResolvedOp::Binary {
                op: OpKind::Add,
                dst: Reg(6),
                left: Reg(4),
                right: Reg(5)
            }
        );
    }

    #[test]
    fn resolve_binary_left_spilled() {
        // left spilled at offset 0, right in v5
        let (assign, spills, remat) = make_maps(&[(1, 5), (2, 6)], &[(0, 0)]);
        let op = ScheduledOp::Binary(OpKind::Add, regalloc::ValueId(0), regalloc::ValueId(1));
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(6)), &assign, &spills, &remat, Reg(13)).unwrap();

        assert_eq!(plan.reloads.len(), 1);
        assert_eq!(
            plan.reloads[0],
            Reload::FromStack {
                target: RELOAD_REGS[1],
                offset: 0
            }
        );
        assert_eq!(
            plan.op,
            ResolvedOp::Binary {
                op: OpKind::Add,
                dst: Reg(6),
                left: RELOAD_REGS[1],
                right: Reg(5)
            }
        );
    }

    #[test]
    fn resolve_binary_both_spilled() {
        // Both spilled: left → dst (temp trick), right → tmp_op
        let (assign, spills, remat) = make_maps(&[(2, 6)], &[(0, 0), (1, 16)]);
        let op = ScheduledOp::Binary(OpKind::Mul, regalloc::ValueId(0), regalloc::ValueId(1));
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(6)), &assign, &spills, &remat, Reg(13)).unwrap();

        assert_eq!(plan.reloads.len(), 2);
        // left → dst (v6), right → tmp_op (v27)
        assert_eq!(
            plan.reloads[0],
            Reload::FromStack {
                target: Reg(6),
                offset: 0
            }
        );
        assert_eq!(
            plan.reloads[1],
            Reload::FromStack {
                target: RELOAD_REGS[1],
                offset: 16
            }
        );
        assert_eq!(
            plan.op,
            ResolvedOp::Binary {
                op: OpKind::Mul,
                dst: Reg(6),
                left: Reg(6),
                right: RELOAD_REGS[1]
            }
        );
    }

    #[test]
    fn resolve_dst_spilled_generates_store() {
        // dst is spilled → compute into RELOAD_REGS[0], then store
        let (assign, spills, remat) = make_maps(&[(0, 4), (1, 5)], &[(2, 32)]);
        let op = ScheduledOp::Binary(OpKind::Add, regalloc::ValueId(0), regalloc::ValueId(1));
        let plan =
            resolve_operands(&op, Loc::Spill(32), &assign, &spills, &remat, Reg(13)).unwrap();

        // dst should be RELOAD_REGS[0] since result is spilled
        assert_eq!(
            plan.op,
            ResolvedOp::Binary {
                op: OpKind::Add,
                dst: RELOAD_REGS[0],
                left: Reg(4),
                right: Reg(5)
            }
        );
        assert_eq!(
            plan.store,
            Some(Store {
                src: RELOAD_REGS[0],
                offset: 32
            })
        );
    }

    #[test]
    fn resolve_muladd_fmla_path() {
        // a in reg, b in reg, c in reg → FMLA with setup_mov for c→dst
        let (assign, spills, remat) = make_maps(&[(0, 4), (1, 5), (2, 7), (3, 8)], &[]);
        let op = ScheduledOp::Ternary(
            OpKind::MulAdd,
            regalloc::ValueId(0),
            regalloc::ValueId(1),
            regalloc::ValueId(2),
        );
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(8)), &assign, &spills, &remat, Reg(13)).unwrap();

        assert!(plan.reloads.is_empty());
        // c=v7 ≠ dst=v8, so setup_mov should copy c → dst
        assert_eq!(plan.setup_mov, Some((Reg(8), Reg(7))));
        assert_eq!(
            plan.op,
            ResolvedOp::FusedMulAdd {
                dst: Reg(8),
                a: Reg(4),
                b: Reg(5)
            }
        );
    }

    #[test]
    fn resolve_muladd_decomposed_both_ab_spilled() {
        // a and b both spilled → decomposed FMUL+FADD path
        // c in register
        let (assign, spills, remat) = make_maps(&[(2, 7), (3, 8)], &[(0, 0), (1, 16)]);
        let op = ScheduledOp::Ternary(
            OpKind::MulAdd,
            regalloc::ValueId(0),
            regalloc::ValueId(1),
            regalloc::ValueId(2),
        );
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(8)), &assign, &spills, &remat, Reg(13)).unwrap();

        // a → dst, b → tmp_op loaded upfront
        assert_eq!(plan.reloads.len(), 2);
        assert_eq!(
            plan.reloads[0],
            Reload::FromStack {
                target: Reg(8),
                offset: 0
            }
        );
        assert_eq!(
            plan.reloads[1],
            Reload::FromStack {
                target: RELOAD_REGS[1],
                offset: 16
            }
        );
        // c is in a register, no deferred reload needed
        match &plan.op {
            ResolvedOp::DecomposedMulAdd {
                dst,
                a,
                b,
                c,
                c_deferred,
            } => {
                assert_eq!(*dst, Reg(8));
                assert_eq!(*a, Reg(8));
                assert_eq!(*b, RELOAD_REGS[1]);
                assert_eq!(*c, Reg(7));
                assert_eq!(*c_deferred, None);
            }
            other => panic!("expected DecomposedMulAdd, got {:?}", other),
        }
    }

    #[test]
    fn resolve_muladd_decomposed_all_three_spilled() {
        // a, b, c all spilled → decomposed with deferred c reload
        let (assign, spills, remat) = make_maps(&[(3, 8)], &[(0, 0), (1, 16), (2, 32)]);
        let op = ScheduledOp::Ternary(
            OpKind::MulAdd,
            regalloc::ValueId(0),
            regalloc::ValueId(1),
            regalloc::ValueId(2),
        );
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(8)), &assign, &spills, &remat, Reg(13)).unwrap();

        // Only a and b reloads upfront — c is deferred
        assert_eq!(plan.reloads.len(), 2);
        match &plan.op {
            ResolvedOp::DecomposedMulAdd { c, c_deferred, .. } => {
                assert_eq!(*c, RELOAD_REGS[1]); // will be reloaded into tmp_op
                assert_eq!(*c_deferred, Some(DeferredReload::FromStack(32)));
            }
            other => panic!("expected DecomposedMulAdd, got {:?}", other),
        }
    }

    #[test]
    fn resolve_var_is_nop() {
        let (assign, spills, remat) = make_maps(&[(0, 0)], &[]);
        let op = ScheduledOp::Var(0);
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(0)), &assign, &spills, &remat, Reg(13)).unwrap();
        assert_eq!(plan.op, ResolvedOp::Nop);
        assert!(plan.reloads.is_empty());
        assert!(plan.store.is_none());
    }

    #[test]
    fn resolve_const() {
        let (assign, spills, remat) = make_maps(&[(0, 6)], &[]);
        let op = ScheduledOp::Const(core::f32::consts::PI);
        let plan =
            resolve_operands(&op, Loc::Reg(Reg(6)), &assign, &spills, &remat, Reg(13)).unwrap();
        assert_eq!(
            plan.op,
            ResolvedOp::LoadConst {
                dst: Reg(6),
                val_bits: core::f32::consts::PI.to_bits()
            }
        );
    }

    // =========================================================================
    // DAG integration tests — expressions that previously crashed (SIGSEGV)
    // =========================================================================

    /// Test that Select short-circuits: when mask is all-true, the false arm
    /// (which contains a division by zero) must NOT produce NaN in the output.
    /// Test Select with all-false mask: should return false arm.
    /// Test Select with mixed mask: BSL path, both arms evaluated.
    // =========================================================================
    // Arena compilation tests
    // =========================================================================

    #[test]
    fn arena_to_schedule_simple() {
        use crate::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let sum = arena.push_binary(OpKind::Add, x, y);

        let schedule = arena_to_schedule(&arena, sum);

        // Should have 3 values: X, Y, X+Y
        assert_eq!(
            schedule.len(),
            3,
            "expected 3 schedule entries, got {}",
            schedule.len()
        );

        // Verify the operations
        assert!(matches!(schedule[0].1, ScheduledOp::Var(0)));
        assert!(matches!(schedule[1].1, ScheduledOp::Var(1)));
        assert!(matches!(
            schedule[2].1,
            ScheduledOp::Binary(OpKind::Add, _, _)
        ));
    }

    #[test]
    fn arena_to_schedule_filters_unreachable() {
        use crate::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let _garbage = arena.push_const(999.0); // unreachable
        let y = arena.push_var(1);
        let sum = arena.push_binary(OpKind::Add, x, y);

        let schedule = arena_to_schedule(&arena, sum);

        // Should have 3 values (garbage node filtered out)
        assert_eq!(
            schedule.len(),
            3,
            "unreachable garbage node should be filtered"
        );
    }

    #[test]
    fn verify_arena_to_uses() {
        use crate::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let sum = arena.push_binary(OpKind::Add, x, y);

        let schedule = arena_to_schedule(&arena, sum);
        let uses = arena_to_uses(&schedule);

        assert_eq!(uses.len(), 3);
        assert!(uses[0].is_empty(), "Var should have no uses");
        assert!(uses[1].is_empty(), "Var should have no uses");
        assert_eq!(uses[2].len(), 2, "Binary should use 2 children");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn arena_compile_simple() {
        use crate::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let sum = arena.push_binary(OpKind::Add, x, y);

        let result = compile_arena_dag(&arena, sum).expect("arena DAG compile failed");
        assert_eq!(result.spill_count, 0);

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(3.0);
            let y = vdupq_n_f32(4.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(out, 0), 7.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn arena_compile_with_constant() {
        use crate::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let two = arena.push_const(2.0);
        let y = arena.push_var(1);
        let prod = arena.push_binary(OpKind::Mul, x, two);
        let sum = arena.push_binary(OpKind::Add, prod, y);

        let result = compile_arena_dag(&arena, sum).expect("arena DAG compile failed");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(3.0);
            let y = vdupq_n_f32(4.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            // 3*2 + 4 = 10
            assert_eq!(vgetq_lane_f32(out, 0), 10.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn arena_compile_with_spills() {
        use crate::arena::ExprArena;

        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let z = arena.push_var(2);
        let w = arena.push_var(3);
        let left = arena.push_binary(OpKind::Add, x, y);
        let right = arena.push_binary(OpKind::Add, z, w);
        let root = arena.push_binary(OpKind::Add, left, right);

        let ctx = EmitCtx::with_max_regs(2);
        let result = compile_arena_dag_with_ctx(&arena, root, ctx)
            .expect("arena DAG compile with spills failed");

        assert!(result.spill_count > 0, "expected spills with max_regs=2");

        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(1.0);
            let y = vdupq_n_f32(2.0);
            let z = vdupq_n_f32(3.0);
            let w = vdupq_n_f32(4.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            // (1+2) + (3+4) = 10
            assert_eq!(vgetq_lane_f32(out, 0), 10.0);
        }
    }

    /// Run a per-batch arena kernel at `x` (Y/Z/W = 0) and return lane 0.
    /// 128-bit `KernelFn`; the builtin-parity tests below use it. Gated off
    /// `+avx512f` (those builtins aren't in the AVX-512 op set yet anyway).
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    fn run1(arena: &ExprArena, root: ExprId, x: f32) -> f32 {
        use core::arch::x86_64::*;
        let r = compile_arena_dag(arena, root).expect("compile failed");
        unsafe {
            let f: executable::KernelFn = r.code.as_fn();
            let z = _mm_set1_ps(0.0);
            _mm_cvtss_f32(f(_mm_set1_ps(x), z, z, z))
        }
    }

    /// Per-batch eval at (X=x, Y=y, Z=W=0), lane 0. 128-bit `KernelFn`, so gated
    /// off `+avx512f` like `run1`.
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    fn run_xy(arena: &ExprArena, root: ExprId, x: f32, y: f32) -> f32 {
        use core::arch::x86_64::*;
        let r = compile_arena_dag(arena, root).expect("compile failed");
        unsafe {
            let f: executable::KernelFn = r.code.as_fn();
            let z = _mm_set1_ps(0.0);
            _mm_cvtss_f32(f(_mm_set1_ps(x), _mm_set1_ps(y), z, z))
        }
    }

    /// A `Dwrt`-carrying arena must JIT-compile end-to-end: the compile entry
    /// runs `lower_dwrt`, so `D(√(x²+y²), x)` compiles to `x / √(x²+y²)`
    /// without the caller ever seeing the derivative machinery.
    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    fn dwrt_compiles_to_analytic_derivative() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let x2 = a.push_binary(OpKind::Mul, x, x);
        let y2 = a.push_binary(OpKind::Mul, y, y);
        let sum = a.push_binary(OpKind::Add, x2, y2);
        let dist = a.push_unary(OpKind::Sqrt, sum);
        let v0 = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, dist, v0);

        for (px, py) in [(3.0f32, 4.0f32), (1.0, 1.0), (-2.0, 5.0)] {
            let got = run_xy(&a, root, px, py);
            let want = px / (px * px + py * py).sqrt();
            assert!(
                (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                "d/dx dist at ({px},{py}): got {got}, want {want}"
            );
        }
    }

    /// A `Dwrt` over an op with no derivative rule must surface as a compile
    /// error (loud refusal), not a miscompile or a scheduler panic.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dwrt_of_gather_refuses_to_compile() {
        use crate::arena::BufferDecl;
        let mut a = ExprArena::new();
        let buf = a.declare_buffer(BufferDecl {
            width: 2,
            height: 1,
        });
        let bufleaf = a.push_buffer(buf);
        let x = a.push_var(0);
        let y = a.push_var(1);
        let g = a.push_ternary(OpKind::Gather, bufleaf, x, y);
        let v0 = a.push_const(0.0);
        let root = a.push_binary(OpKind::Dwrt, g, v0);
        assert!(compile_arena_dag(&a, root).is_err());
    }

    /// A spill frame past the 128-byte red zone must allocate a real frame
    /// (`sub rsp`) and produce correct results — the glyph-scale-kernel case
    /// that used to refuse with "exceeds 128-byte red zone". 40 products are
    /// all pushed before any is consumed, so dozens are simultaneously live
    /// against 6 allocatable registers.
    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    fn spill_frame_beyond_red_zone_compiles_correctly() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let mut products = alloc::vec::Vec::new();
        for i in 0..40u32 {
            let c = a.push_const(i as f32 + 1.0);
            let xa = a.push_binary(OpKind::Add, x, c);
            let yb = a.push_binary(OpKind::Add, y, c);
            products.push(a.push_binary(OpKind::Mul, xa, yb));
        }
        let mut root = products[0];
        for p in &products[1..] {
            root = a.push_binary(OpKind::Add, root, *p);
        }

        let result = compile_arena_dag(&a, root).expect("large spill frame must compile");
        assert!(
            result.spill_bytes > 128,
            "test did not force a frame beyond the red zone (spill_bytes = {})",
            result.spill_bytes
        );

        for (px, py) in [(1.5f32, -2.0f32), (0.0, 0.0), (3.0, 4.0)] {
            let got = run_xy(&a, root, px, py);
            let want: f32 = (0..40)
                .map(|i| (px + i as f32 + 1.0) * (py + i as f32 + 1.0))
                .sum();
            let tol = 1e-3 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "at ({px},{py}): jit {got}, scalar {want}"
            );
        }
    }

    /// Every x86-64 unary transcendental/round op must match its scalar
    /// reference across a range of inputs — these exercise `emit_arena` →
    /// `emit_unary` directly (not the compiler's lowering).
    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    fn x86_unary_builtins_match_scalar() {
        // Tolerances reflect the shared (with aarch64) minimax-polynomial
        // accuracy over a sensible input range; exact ops use tight bounds.
        // `rel_err = |jit - scalar| / (1 + |scalar|)`.
        type UnaryCase<'a> = (OpKind, fn(f32) -> f32, &'a [f32], f32);
        let unary: &[UnaryCase] = &[
            (
                OpKind::Sqrt,
                |x| x.sqrt(),
                &[0.25, 1.0, 2.0, 9.0, 100.0],
                1e-5,
            ),
            (OpKind::Abs, |x| x.abs(), &[-3.0, -0.5, 0.0, 2.5], 1e-6),
            (OpKind::Neg, |x| -x, &[-3.0, 0.5, 2.5], 1e-6),
            (
                OpKind::Floor,
                |x| x.floor(),
                &[-2.3, -0.1, 0.9, 1.5, 3.99],
                1e-6,
            ),
            (
                OpKind::Ceil,
                |x| x.ceil(),
                &[-2.3, -0.1, 0.9, 1.5, 3.01],
                1e-6,
            ),
            (
                OpKind::Round,
                |x| x.round_ties_even(),
                &[-2.4, -0.4, 0.4, 1.5, 2.6],
                1e-6,
            ),
            // sin/cos: 4-term Chebyshev — accurate well inside [-π, π].
            (
                OpKind::Sin,
                |x| x.sin(),
                &[-2.0, -1.0, -0.3, 0.0, 0.5, 1.5, 2.0],
                6e-3,
            ),
            (
                OpKind::Cos,
                |x| x.cos(),
                &[-1.0, -0.3, 0.0, 0.5, 1.0],
                1.5e-2,
            ),
            (
                OpKind::Tan,
                |x| x.tan(),
                &[-1.0, -0.3, 0.0, 0.3, 1.0],
                2.5e-2,
            ),
            (
                OpKind::Exp,
                |x| x.exp(),
                &[-2.0, -0.5, 0.0, 1.0, 2.0, 3.0],
                5e-3,
            ),
            (
                OpKind::Exp2,
                |x| x.exp2(),
                &[-3.0, -0.5, 0.0, 1.0, 4.0],
                5e-3,
            ),
            (OpKind::Ln, |x| x.ln(), &[0.25, 0.5, 1.0, 2.0, 10.0], 5e-3),
            (
                OpKind::Log2,
                |x| x.log2(),
                &[0.25, 0.5, 1.0, 2.0, 8.0],
                5e-3,
            ),
            (
                OpKind::Log10,
                |x| x.log10(),
                &[0.1, 0.5, 1.0, 10.0, 100.0],
                5e-3,
            ),
            (
                OpKind::Atan,
                |x| x.atan(),
                &[-5.0, -0.5, -0.2, 0.0, 0.2, 0.5, 5.0],
                8e-3,
            ),
            (
                OpKind::Asin,
                |x| x.asin(),
                &[-0.8, -0.5, 0.0, 0.5, 0.8],
                1e-2,
            ),
            (
                OpKind::Acos,
                |x| x.acos(),
                &[-0.8, -0.5, 0.0, 0.5, 0.8],
                1e-2,
            ),
        ];
        for &(op, scalar, inputs, tol) in unary {
            let mut arena = ExprArena::new();
            let x = arena.push_var(0);
            let root = arena.push_unary(op, x);
            for &xv in inputs {
                let got = run1(&arena, root, xv);
                let want = scalar(xv);
                let err = (got - want).abs() / (1.0 + want.abs());
                assert!(
                    err <= tol,
                    "{op:?}({xv}): jit={got} scalar={want} rel_err={err} > {tol}"
                );
            }
        }
    }

    /// Binary transcendentals + comparisons + ternaries, JIT vs scalar.
    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    fn x86_binary_ternary_builtins_match_scalar() {
        use core::arch::x86_64::*;
        // Helper: compile f(X, Y) and eval at (x, y).
        unsafe fn run2(arena: &ExprArena, root: ExprId, x: f32, y: f32) -> f32 {
            unsafe {
                let r = compile_arena_dag(arena, root).expect("compile failed");
                let f: executable::KernelFn = r.code.as_fn();
                let z = _mm_set1_ps(0.0);
                _mm_cvtss_f32(f(_mm_set1_ps(x), _mm_set1_ps(y), z, z))
            }
        }

        // atan2(y, x): arena Binary(Atan2, Y, X)  (op order: src1=y, src2=x)
        let pts = [
            (0.5, 2.0),
            (2.0, 0.5),
            (-0.5, 2.0),
            (0.5, -2.0),
            (-2.0, -0.5),
            (3.0, -0.5),
        ];
        {
            let mut a = ExprArena::new();
            let y = a.push_var(1);
            let x = a.push_var(0);
            let root = a.push_binary(OpKind::Atan2, y, x);
            for &(yv, xv) in &pts {
                let got = unsafe { run2(&a, root, xv, yv) };
                let want = yv.atan2(xv);
                assert!(
                    (got - want).abs() <= 1.5e-2,
                    "atan2({yv},{xv}): {got} vs {want}"
                );
            }
        }
        // pow(X, Y)
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let root = a.push_binary(OpKind::Pow, x, y);
            for &(xv, yv) in &[(2.0f32, 3.0f32), (9.0, 0.5), (4.0, -1.0), (1.5, 2.0)] {
                let got = unsafe { run2(&a, root, xv, yv) };
                let want = xv.powf(yv);
                let err = (got - want).abs() / (1.0 + want.abs());
                assert!(err <= 5e-3, "pow({xv},{yv}): {got} vs {want} err={err}");
            }
        }
        // hypot(X, Y) — the sqrt(x² + y²) composition it denotes.
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let xx = a.push_binary(OpKind::Mul, x, x);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let sum = a.push_binary(OpKind::Add, xx, yy);
            let root = a.push_unary(OpKind::Sqrt, sum);
            for &(xv, yv) in &[(3.0f32, 4.0f32), (1.0, 1.0), (0.0, 2.0)] {
                let got = unsafe { run2(&a, root, xv, yv) };
                let want = xv.hypot(yv);
                assert!(
                    (got - want).abs() <= 1e-4,
                    "hypot({xv},{yv}): {got} vs {want}"
                );
            }
        }
        // Min / Max
        for (op, f) in [
            (OpKind::Min, f32::min as fn(f32, f32) -> f32),
            (OpKind::Max, f32::max as fn(f32, f32) -> f32),
        ] {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let root = a.push_binary(op, x, y);
            for &(xv, yv) in &[(1.0f32, 2.0f32), (3.0, -1.0), (-2.0, -5.0)] {
                let got = unsafe { run2(&a, root, xv, yv) };
                assert!((got - f(xv, yv)).abs() <= 1e-6, "{op:?}({xv},{yv})");
            }
        }
        // clamp(X, 0.0, 1.0) — the min/max composition it denotes.
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let lo = a.push_const(0.0);
            let hi = a.push_const(1.0);
            let floored = a.push_binary(OpKind::Max, x, lo);
            let root = a.push_binary(OpKind::Min, floored, hi);
            for &xv in &[-0.5f32, 0.25, 0.9, 1.7] {
                let got = run1(&a, root, xv);
                assert!(
                    (got - xv.clamp(0.0, 1.0)).abs() <= 1e-6,
                    "clamp({xv})={got}"
                );
            }
        }
        // Select(X >= 0, 1.0, -1.0) == signum-ish
        {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let zero = a.push_const(0.0);
            let cond = a.push_binary(OpKind::Ge, x, zero);
            let pos = a.push_const(1.0);
            let neg = a.push_const(-1.0);
            let root = a.push_ternary(OpKind::Select, cond, pos, neg);
            for &xv in &[-2.0f32, -0.1, 0.1, 3.0] {
                let got = run1(&a, root, xv);
                let want = if xv >= 0.0 { 1.0 } else { -1.0 };
                assert!((got - want).abs() <= 1e-6, "select({xv})={got} want={want}");
            }
        }
    }

    // =========================================================================
    // Forward-mode dual (jet) lowering — validated against analytic derivatives.
    // Uses hardware sqrtps/divps (no polynomial approximations), so tolerances
    // are tight.
    /// Transcendental lowering: sin/cos/tan JIT through the per-batch path with
    /// no backend ever emitting a transcendental (they expand to arithmetic in
    /// `lowering`). Validated against `f32` on the default (128-bit) build.
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    mod lowering_tests {
        use super::*;
        use crate::arena::ExprArena;

        // Tolerance reflects the degree-7 Chebyshev's ACTUAL measured accuracy
        // (the SAME polynomial the runtime Compounds path uses): ~1e-6 near 0,
        // degrading to ~2.6e-2 at the ±π range-reduction edges — only ~2-digit
        // accurate there. The bound catches *logic* errors (sign/coefficient/
        // range-reduction), not approximation error; tightening the polynomial
        // is the tunable-precision lever, applied in one place (`lowering`),
        // later.
        const TRIG_TOL: f32 = 3e-2;

        #[test]
        fn sin_cos_tan_match_scalar() {
            // Range beyond [-π,π] to exercise the floor-based range reduction.
            let pts = [0.0f32, 0.3, 1.0, 2.0, 3.5, -1.7, 6.0, -4.2];
            for &xv in &pts {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let s = a.push_unary(OpKind::Sin, x);
                assert!((run1(&a, s, xv) - xv.sin()).abs() <= TRIG_TOL, "sin({xv})");

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let c = a.push_unary(OpKind::Cos, x);
                assert!((run1(&a, c, xv) - xv.cos()).abs() <= TRIG_TOL, "cos({xv})");
            }
            // tan away from its poles (ratio of two ~3e-3 approximations).
            for &xv in &[0.0f32, 0.3, 0.7, -0.5, 1.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let t = a.push_unary(OpKind::Tan, x);
                // tan = sin/cos amplifies cos's ~2e-2 edge error as |x| grows
                // (measured ~3.8e-2 at x=1). Honest bound for this polynomial.
                assert!((run1(&a, t, xv) - xv.tan()).abs() <= 5e-2, "tan({xv})");
            }
        }

        /// exp/exp2/ln/log2/log10 lower to arithmetic via the bit-manip
        /// primitives (TruncToInt/IntToFloat/IAdd/Shl/Shr/BitAnd/BitOr) — the
        /// float↔int twiddling no backend can avoid. Validated vs `f32`.
        #[test]
        fn exp_log_match_scalar() {
            // exp / exp2 over a moderate range.
            for &xv in &[-2.0f32, -0.5, 0.0, 0.7, 1.5, 3.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let e = a.push_unary(OpKind::Exp, x);
                let rel = (run1(&a, e, xv) - xv.exp()).abs() / xv.exp().max(1.0);
                assert!(rel <= 1e-2, "exp({xv})");

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let e2 = a.push_unary(OpKind::Exp2, x);
                let rel = (run1(&a, e2, xv) - xv.exp2()).abs() / xv.exp2().max(1.0);
                assert!(rel <= 1e-2, "exp2({xv})");
            }
            // ln / log2 / log10 over positive inputs.
            for &xv in &[0.25f32, 0.5, 1.0, 2.0, 5.0, 100.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let l = a.push_unary(OpKind::Ln, x);
                assert!((run1(&a, l, xv) - xv.ln()).abs() <= 3e-2, "ln({xv})");

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let l2 = a.push_unary(OpKind::Log2, x);
                assert!((run1(&a, l2, xv) - xv.log2()).abs() <= 3e-2, "log2({xv})");
            }
        }

        /// atan/atan2/asin/acos lower to arithmetic + Select (atan2 is the core;
        /// the others derive from it). Value path only — atan2 uses Select, which
        /// the jet path can't differentiate. Validated vs `f32`.
        #[test]
        fn inverse_trig_match_scalar() {
            // The degree-7 atan polynomial is worst (~6e-2) at |ratio|=1; that
            // dominates the tolerance. It catches logic/quadrant errors (which
            // were off by whole radians via the guard bug), not approximation
            // error. Tightening the polynomial is the tunable-precision lever.
            const ATAN_TOL: f32 = 7e-2;

            // atan over a wide range (exercises the |ratio|>1 swap branch).
            for &xv in &[0.0f32, 0.3, 1.0, 2.5, -0.7, -4.0] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let at = a.push_unary(OpKind::Atan, x);
                assert!(
                    (run1(&a, at, xv) - xv.atan()).abs() <= ATAN_TOL,
                    "atan({xv})"
                );
            }
            // asin/acos on [-1, 1].
            for &xv in &[-0.9f32, -0.4, 0.0, 0.4, 0.9] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let s = a.push_unary(OpKind::Asin, x);
                assert!(
                    (run1(&a, s, xv) - xv.asin()).abs() <= ATAN_TOL,
                    "asin({xv})"
                );

                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let c = a.push_unary(OpKind::Acos, x);
                assert!(
                    (run1(&a, c, xv) - xv.acos()).abs() <= ATAN_TOL,
                    "acos({xv})"
                );
            }
            // atan2 across quadrants (y in var0, x in var1). The (1,1)/(-1,-1)…
            // cases sit at |ratio|=1, the polynomial's worst point.
            let pts = [
                (1.0f32, 1.0f32),
                (1.0, -1.0),
                (-1.0, -1.0),
                (-1.0, 1.0),
                (0.5, -2.0),
            ];
            for &(yv, xv) in &pts {
                let mut a = ExprArena::new();
                let y = a.push_var(0);
                let x = a.push_var(1);
                let r = a.push_binary(OpKind::Atan2, y, x);
                let got = run_xy(&a, r, yv, xv);
                assert!(
                    (got - yv.atan2(xv)).abs() <= ATAN_TOL,
                    "atan2({yv},{xv}) = {got}"
                );
            }
        }

        /// A transcendental composed inside arithmetic still works: sin(x)·x + 1.
        #[test]
        fn transcendental_in_expression() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let s = a.push_unary(OpKind::Sin, x);
            let sx = a.push_binary(OpKind::Mul, s, x);
            let one = a.push_const(1.0);
            let root = a.push_binary(OpKind::Add, sx, one);
            for &xv in &[0.2f32, 0.9, 2.1, -1.3] {
                let want = xv.sin() * xv + 1.0;
                // sin's ~3e-3 error is scaled by |x|, so allow for that.
                let tol = 3e-3 * (1.0 + xv.abs());
                assert!(
                    (run1(&a, root, xv) - want).abs() <= tol,
                    "sin(x)·x+1 @ {xv}"
                );
            }
        }
    }

    // =========================================================================
    // x86 shared-pipeline per-batch path (schedule → regalloc → spill).
    // =========================================================================
    // Calls kernels through the per-batch `KernelFn` (128-bit here); gated off
    // `+avx512f` where that ABI is `__m512` (AVX-512 covered by `avx512_driver`).
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    mod sched {
        use super::*;
        use crate::arena::ExprArena;

        fn run(res: &CompileResult, x: f32, y: f32, z: f32, w: f32) -> f32 {
            unsafe {
                use core::arch::x86_64::*;
                let f: executable::KernelFn = res.code.as_fn();
                let o = f(
                    _mm_set1_ps(x),
                    _mm_set1_ps(y),
                    _mm_set1_ps(z),
                    _mm_set1_ps(w),
                );
                _mm_cvtss_f32(o)
            }
        }

        const PTS: &[(f32, f32, f32, f32)] = &[
            (3.0, 4.0, 0.0, 1.0),
            (1.0, 2.0, 3.0, 4.0),
            (-2.0, 0.5, 1.5, -1.0),
            (0.7, -1.3, 2.1, 0.2),
        ];

        /// The scheduled path must agree with the Sethi-Ullman path (and ground
        /// truth) for expressions that fit in registers.
        #[test]
        fn sched_parity_no_spill() {
            // f = sqrt(X*X + Y*Y) - Z, plus a non-commutative `X - Y*Z` shape.
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let z = a.push_var(2);
            let xx = a.push_binary(OpKind::Mul, x, x);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let sum = a.push_binary(OpKind::Add, xx, yy);
            let dist = a.push_unary(OpKind::Sqrt, sum);
            let yz = a.push_binary(OpKind::Mul, y, z);
            let sub = a.push_binary(OpKind::Sub, dist, yz); // dist - Y*Z
            let root = sub;

            let sethi = compile_arena_dag(&a, root).expect("sethi compile");
            let sched = compile_arena_dag(&a, root).expect("scheduled compile");
            assert_eq!(sched.spill_count, 0, "should fit without spilling");

            for &(px, py, pz, pw) in PTS {
                let want = (px * px + py * py).sqrt() - py * pz;
                let g_sethi = run(&sethi, px, py, pz, pw);
                let g_sched = run(&sched, px, py, pz, pw);
                assert!(
                    (g_sethi - want).abs() <= 1e-4,
                    "sethi {g_sethi} want {want}"
                );
                assert!(
                    (g_sched - want).abs() <= 1e-4,
                    "sched {g_sched} want {want}"
                );
            }
        }

        /// A wide expression that exceeds the 7 allocatable registers must spill
        /// (to the red zone) and still compute the right answer.
        #[test]
        fn sched_spills_and_is_correct() {
            // sum_{i=1..=10} (X + i) * (Y + i), as a balanced tree so the 10
            // products are live together — forcing spills with only 7 regs.
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let mut terms = alloc::vec::Vec::new();
            for i in 1..=10u32 {
                let c = a.push_const(i as f32);
                let ax = a.push_binary(OpKind::Add, x, c);
                let by = a.push_binary(OpKind::Add, y, c);
                terms.push(a.push_binary(OpKind::Mul, ax, by));
            }
            while terms.len() > 1 {
                let mut next = alloc::vec::Vec::new();
                let it = terms.chunks(2);
                for pair in it {
                    if pair.len() == 2 {
                        next.push(a.push_binary(OpKind::Add, pair[0], pair[1]));
                    } else {
                        next.push(pair[0]);
                    }
                }
                terms = next;
            }
            let root = terms[0];

            let sched = compile_arena_dag(&a, root).expect("scheduled compile");
            assert!(
                sched.spill_count > 0,
                "expected spilling; widen the expression if this regresses"
            );

            for &(px, py, _pz, _pw) in PTS {
                let mut want = 0.0f32;
                for i in 1..=10u32 {
                    want += (px + i as f32) * (py + i as f32);
                }
                let got = run(&sched, px, py, 0.0, 0.0);
                let tol = 1e-3 * want.abs().max(1.0);
                assert!((got - want).abs() <= tol, "spill: got {got} want {want}");
            }
        }

        /// Exercises the shared driver's Select short-circuit guard path on x86
        /// (MOVMSKPS all-true/all-false branches): `(X > 0) ? Y*Y*Y : Z+Z+Z`,
        /// with arm-exclusive subexpressions so a guard region forms. Uniform
        /// inputs take the all-true / all-false branches.
        #[test]
        fn sched_select_guards() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let z = a.push_var(2);
            let zero = a.push_const(0.0);
            let cond = a.push_binary(OpKind::Gt, x, zero); // X > 0 -> mask
            let yy = a.push_binary(OpKind::Mul, y, y);
            let yyy = a.push_binary(OpKind::Mul, yy, y); // true arm: Y^3
            let zz = a.push_binary(OpKind::Add, z, z);
            let zzz = a.push_binary(OpKind::Add, zz, z); // false arm: 3Z
            let root = a.push_ternary(OpKind::Select, cond, yyy, zzz);

            let sched = compile_arena_dag(&a, root).expect("scheduled compile");

            // x>0 -> all-true -> Y^3 ; x<=0 -> all-false -> 3Z.
            for &(px, py, pz, _pw) in PTS {
                let want = if px > 0.0 { py * py * py } else { 3.0 * pz };
                let got = run(&sched, px, py, pz, 0.0);
                assert!(
                    (got - want).abs() <= 1e-3,
                    "select: ({px},{py},{pz}) got {got} want {want}"
                );
            }
        }
    }

    // =========================================================================
    // 128-bit end-to-end: bound-memory gather through the shared driver, run on
    // the host across 4 lanes. Covers BOTH 128-bit backends — NEON's native
    // `ld1` lanes and x86's scalar-load assembly (no AVX2 `vgatherdps` at 128
    // bits) — against the same interpreter oracle, so the two cannot drift.
    // Mirrors the avx512_driver gather tests at 128-bit width.
    // =========================================================================
    #[cfg(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            not(target_feature = "avx512f"),
            not(target_feature = "avx2")
        )
    ))]
    mod gather_driver_128 {
        use super::*;
        use crate::arena::{ExprArena, ExprId};

        /// Run a compiled gather kernel as a `CtxKernelFn`: the context (array
        /// of buffer base pointers) goes in the first integer argument (x0 /
        /// rdi), coords in the first four vector registers.
        #[cfg(target_arch = "aarch64")]
        fn run4_ctx(
            res: &CompileResult,
            ctx: &[*const f32],
            xs: [f32; 4],
            ys: [f32; 4],
        ) -> [f32; 4] {
            unsafe {
                use core::arch::aarch64::*;
                let f: executable::CtxKernelFn = res.code.as_fn();
                let r = f(
                    ctx.as_ptr(),
                    vld1q_f32(xs.as_ptr()),
                    vld1q_f32(ys.as_ptr()),
                    vdupq_n_f32(0.0),
                    vdupq_n_f32(0.0),
                );
                let mut out = [0.0f32; 4];
                vst1q_f32(out.as_mut_ptr(), r);
                out
            }
        }

        /// See the aarch64 variant above.
        #[cfg(target_arch = "x86_64")]
        fn run4_ctx(
            res: &CompileResult,
            ctx: &[*const f32],
            xs: [f32; 4],
            ys: [f32; 4],
        ) -> [f32; 4] {
            unsafe {
                use core::arch::x86_64::*;
                let f: executable::CtxKernelFn = res.code.as_fn();
                let r = f(
                    ctx.as_ptr(),
                    _mm_loadu_ps(xs.as_ptr()),
                    _mm_loadu_ps(ys.as_ptr()),
                    _mm_setzero_ps(),
                    _mm_setzero_ps(),
                );
                let mut out = [0.0f32; 4];
                _mm_storeu_ps(out.as_mut_ptr(), r);
                out
            }
        }

        #[allow(clippy::too_many_arguments)] // test helper: 6 distinct params (arena, root, buffers, xs, ys, tag)
        /// Check a compiled gather kernel lane-for-lane against `eval_scalar`,
        /// the reference interpreter, over the same coords and binding. The 16
        /// coordinate pairs run as four 4-lane batches.
        fn check_against_interp(
            arena: &ExprArena,
            root: ExprId,
            buffers: &[&[f32]],
            xs: [f32; 16],
            ys: [f32; 16],
            tag: &str,
        ) {
            let res = compile_arena_dag(arena, root).expect("compile gather kernel");
            let ctx: Vec<*const f32> = buffers.iter().map(|b| b.as_ptr()).collect();
            let bindings = crate::binding::BindingTable::bind(arena, buffers).unwrap();

            for batch in 0..4 {
                let mut cx = [0.0f32; 4];
                let mut cy = [0.0f32; 4];
                cx.copy_from_slice(&xs[batch * 4..batch * 4 + 4]);
                cy.copy_from_slice(&ys[batch * 4..batch * 4 + 4]);
                let got = run4_ctx(&res, &ctx, cx, cy);
                for i in 0..4 {
                    let want =
                        crate::eval::eval_scalar(arena, root, &[cx[i], cy[i], 0.0, 0.0], &bindings);
                    assert_eq!(
                        got[i], want,
                        "{tag} batch {batch} lane {i} (x={}, y={})",
                        cx[i], cy[i]
                    );
                }
            }
        }

        fn idx_lanes() -> ([f32; 16], [f32; 16]) {
            // A spread of in-range, fractional, and out-of-range coordinates so
            // the clamp and floor paths are all exercised.
            let xs = [
                0.0, 1.0, 2.9, 7.0, -3.0, 100.0, 4.0, 5.5, 6.0, 0.1, 3.0, 2.0, 1.9, 7.9, -0.5, 4.4,
            ];
            let ys = [
                0.0, 0.0, 1.0, 1.9, 2.0, 2.0, -1.0, 3.0, 0.5, 2.9, 1.0, 3.9, 0.0, 2.0, 5.0, 1.0,
            ];
            (xs, ys)
        }

        #[test]
        fn gather_jit_matches_interpreter() {
            // 8x4 buffer, gather at (X, Y).
            let (w, h) = (8usize, 4usize);
            let buf: Vec<f32> = (0..(w * h)).map(|i| i as f32 * 2.0 - 3.0).collect();
            let mut a = ExprArena::new();
            let b = a.declare_buffer(crate::arena::BufferDecl {
                width: w as u32,
                height: h as u32,
            });
            let x = a.push_var(0);
            let y = a.push_var(1);
            let root = a.push_gather(b, x, y);
            let (xs, ys) = idx_lanes();
            check_against_interp(&a, root, &[buf.as_slice()], xs, ys, "gather");
        }

        #[test]
        fn gather_composed_with_arithmetic() {
            // out = gather(buf, X, Y) * 2 + Y — proves gather is a schedulable
            // mid-expression node, not just a whole-kernel root.
            let (w, h) = (8usize, 4usize);
            let buf: Vec<f32> = (0..(w * h)).map(|i| (i as f32).sin()).collect();
            let mut a = ExprArena::new();
            let b = a.declare_buffer(crate::arena::BufferDecl {
                width: w as u32,
                height: h as u32,
            });
            let x = a.push_var(0);
            let y = a.push_var(1);
            let g = a.push_gather(b, x, y);
            let two = a.push_const(2.0);
            let scaled = a.push_binary(OpKind::Mul, g, two);
            let root = a.push_binary(OpKind::Add, scaled, y);
            let (xs, ys) = idx_lanes();
            check_against_interp(&a, root, &[buf.as_slice()], xs, ys, "gather*2+Y");
        }

        #[test]
        fn gather_two_buffers() {
            // gA(X,Y) + gB(Y,X) with two distinct bound buffers, exercising
            // slot 0 and slot 1 of the context.
            let (w, h) = (6usize, 6usize);
            let buf_a: Vec<f32> = (0..(w * h)).map(|i| i as f32).collect();
            let buf_b: Vec<f32> = (0..(w * h)).map(|i| -(i as f32) * 0.5).collect();
            let mut a = ExprArena::new();
            let ba = a.declare_buffer(crate::arena::BufferDecl {
                width: w as u32,
                height: h as u32,
            });
            let bb = a.declare_buffer(crate::arena::BufferDecl {
                width: w as u32,
                height: h as u32,
            });
            let x = a.push_var(0);
            let y = a.push_var(1);
            let ga = a.push_gather(ba, x, y);
            let gb = a.push_gather(bb, y, x);
            let root = a.push_binary(OpKind::Add, ga, gb);
            let (xs, ys) = idx_lanes();
            check_against_interp(
                &a,
                root,
                &[buf_a.as_slice(), buf_b.as_slice()],
                xs,
                ys,
                "2-buf",
            );
        }

        #[test]
        fn matmul_reduce_jit_matches_interpreter() {
            // out(j) = Σ_i W(i,j) * input(i), evaluated per output lane j = X.
            // The reduction over i unrolls to a flat gather/FMA chain (bound
            // extent), and the whole thing runs as one CtxKernelFn.
            //   W is IN×OUT row-major (width=IN, height=OUT); input is length IN.
            let (in_dim, out_dim) = (4usize, 6usize);
            let w: Vec<f32> = (0..(in_dim * out_dim))
                .map(|k| (k as f32) * 0.5 - 2.0)
                .collect();
            let input: Vec<f32> = (0..in_dim).map(|k| k as f32 + 1.0).collect();

            let mut a = ExprArena::new();
            let wb = a.declare_buffer(crate::arena::BufferDecl {
                width: in_dim as u32,
                height: out_dim as u32,
            });
            let ib = a.declare_buffer(crate::arena::BufferDecl {
                width: in_dim as u32,
                height: 1,
            });
            // body(i, j=X) = W(i, X) * input(i, 0)
            let i = a.push_var(4);
            let j = a.push_var(0);
            let zero = a.push_const(0.0);
            let wg = a.push_gather(wb, i, j);
            let ig = a.push_gather(ib, i, zero);
            let prod = a.push_binary(OpKind::Mul, wg, ig);
            let root = a.push_reduce(OpKind::Add, 4, in_dim as u32, prod);

            let buffers: &[&[f32]] = &[w.as_slice(), input.as_slice()];
            // Output lanes j = 0..6 (rest clamp to the last row, harmless here).
            let xs = [
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0,
            ];
            let ys = [0.0f32; 16];
            check_against_interp(&a, root, buffers, xs, ys, "matmul");
        }
    }

    // =========================================================================
    // AVX-512 end-to-end: arena -> shared driver -> EVEX zmm kernel, run on the
    // host across all 16 lanes. Built only with +avx512f.
    // =========================================================================
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    mod avx512_driver {
        use super::*;
        use crate::arena::ExprArena;

        // Passing __m512 by value IS the emitted ABI (SysV: zmm0-7), so
        // not-FFI-safe is a false positive here, as for `executable`'s aliases.
        #[allow(improper_ctypes_definitions)]
        type K = unsafe extern "C" fn(
            core::arch::x86_64::__m512,
            core::arch::x86_64::__m512,
            core::arch::x86_64::__m512,
            core::arch::x86_64::__m512,
        ) -> core::arch::x86_64::__m512;

        /// Run a compiled zmm kernel over 16 distinct lanes per coordinate.
        fn run16(res: &CompileResult, xs: [f32; 16], ys: [f32; 16], zs: [f32; 16]) -> [f32; 16] {
            unsafe {
                use core::arch::x86_64::*;
                let f: K = res.code.as_fn();
                let r = f(
                    _mm512_loadu_ps(xs.as_ptr()),
                    _mm512_loadu_ps(ys.as_ptr()),
                    _mm512_loadu_ps(zs.as_ptr()),
                    _mm512_setzero_ps(),
                );
                let mut out = [0.0f32; 16];
                _mm512_storeu_ps(out.as_mut_ptr(), r);
                out
            }
        }

        fn lanes() -> ([f32; 16], [f32; 16], [f32; 16]) {
            let mut xs = [0.0; 16];
            let mut ys = [0.0; 16];
            let mut zs = [0.0; 16];
            for i in 0..16 {
                xs[i] = i as f32 - 7.0;
                ys[i] = (i as f32) * 0.5 + 1.0;
                zs[i] = 3.0 - (i as f32) * 0.25;
            }
            (xs, ys, zs)
        }

        fn check(got: [f32; 16], want: impl Fn(usize) -> f32, tag: &str) {
            for (i, &g) in got.iter().enumerate() {
                let w = want(i);
                assert!(
                    (g - w).abs() <= 1e-3,
                    "{tag} lane {i}: got {} want {}",
                    g,
                    w
                );
            }
        }

        // ---- Bound-memory gather: JIT vs reference interpreter ----

        // Passing __m512 by value IS the emitted ABI (SysV: zmm0-7), so
        // not-FFI-safe is a false positive here, as for `executable`'s aliases.
        #[allow(improper_ctypes_definitions)]
        type CtxK = unsafe extern "C" fn(
            *const *const f32,
            core::arch::x86_64::__m512,
            core::arch::x86_64::__m512,
            core::arch::x86_64::__m512,
            core::arch::x86_64::__m512,
        ) -> core::arch::x86_64::__m512;

        /// Run a compiled gather kernel as a `CtxKernelFn`: the context (array of
        /// buffer base pointers) goes in rdi, coords in zmm0..3.
        fn run16_ctx(
            res: &CompileResult,
            ctx: &[*const f32],
            xs: [f32; 16],
            ys: [f32; 16],
        ) -> [f32; 16] {
            unsafe {
                use core::arch::x86_64::*;
                let f: CtxK = res.code.as_fn();
                let r = f(
                    ctx.as_ptr(),
                    _mm512_loadu_ps(xs.as_ptr()),
                    _mm512_loadu_ps(ys.as_ptr()),
                    _mm512_setzero_ps(),
                    _mm512_setzero_ps(),
                );
                let mut out = [0.0f32; 16];
                _mm512_storeu_ps(out.as_mut_ptr(), r);
                out
            }
        }

        /// Check a compiled gather kernel lane-for-lane against `eval_scalar`,
        /// the reference interpreter, over the same coords and binding.
        #[allow(clippy::too_many_arguments)] // test helper: 6 distinct params (arena, root, buffers, xs, ys, tag)
        fn check_against_interp(
            arena: &ExprArena,
            root: ExprId,
            buffers: &[&[f32]],
            xs: [f32; 16],
            ys: [f32; 16],
            tag: &str,
        ) {
            let res = compile_arena_dag(arena, root).expect("compile gather kernel");
            let ctx: Vec<*const f32> = buffers.iter().map(|b| b.as_ptr()).collect();
            let got = run16_ctx(&res, &ctx, xs, ys);

            let bindings = crate::binding::BindingTable::bind(arena, buffers).unwrap();
            for (i, &g) in got.iter().enumerate() {
                let want =
                    crate::eval::eval_scalar(arena, root, &[xs[i], ys[i], 0.0, 0.0], &bindings);
                assert_eq!(g, want, "{tag} lane {i} (x={}, y={})", xs[i], ys[i]);
            }
        }

        fn idx_lanes() -> ([f32; 16], [f32; 16]) {
            // A spread of in-range, fractional, and out-of-range coordinates so
            // the clamp and floor paths are all exercised.
            let xs = [
                0.0, 1.0, 2.9, 7.0, -3.0, 100.0, 4.0, 5.5, 6.0, 0.1, 3.0, 2.0, 1.9, 7.9, -0.5, 4.4,
            ];
            let ys = [
                0.0, 0.0, 1.0, 1.9, 2.0, 2.0, -1.0, 3.0, 0.5, 2.9, 1.0, 3.9, 0.0, 2.0, 5.0, 1.0,
            ];
            (xs, ys)
        }

        #[test]
        fn gather_jit_matches_interpreter() {
            // 8x4 buffer, gather at (X, Y).
            let (w, h) = (8usize, 4usize);
            let buf: Vec<f32> = (0..(w * h)).map(|i| i as f32 * 2.0 - 3.0).collect();
            let mut a = ExprArena::new();
            let b = a.declare_buffer(crate::arena::BufferDecl {
                width: w as u32,
                height: h as u32,
            });
            let x = a.push_var(0);
            let y = a.push_var(1);
            let root = a.push_gather(b, x, y);
            let (xs, ys) = idx_lanes();
            check_against_interp(&a, root, &[buf.as_slice()], xs, ys, "gather");
        }

        #[test]
        fn gather_composed_with_arithmetic() {
            // out = gather(buf, X, Y) * 2 + Y — proves gather is a schedulable
            // mid-expression node, not just a whole-kernel root.
            let (w, h) = (8usize, 4usize);
            let buf: Vec<f32> = (0..(w * h)).map(|i| (i as f32).sin()).collect();
            let mut a = ExprArena::new();
            let b = a.declare_buffer(crate::arena::BufferDecl {
                width: w as u32,
                height: h as u32,
            });
            let x = a.push_var(0);
            let y = a.push_var(1);
            let g = a.push_gather(b, x, y);
            let two = a.push_const(2.0);
            let scaled = a.push_binary(OpKind::Mul, g, two);
            let root = a.push_binary(OpKind::Add, scaled, y);
            let (xs, ys) = idx_lanes();
            check_against_interp(&a, root, &[buf.as_slice()], xs, ys, "gather*2+Y");
        }

        #[test]
        fn gather_two_buffers() {
            // coverage.select via arithmetic: gA(X,Y) + gB(Y,X) with two distinct
            // bound buffers, exercising slot 0 and slot 1 of the context.
            let (w, h) = (6usize, 6usize);
            let buf_a: Vec<f32> = (0..(w * h)).map(|i| i as f32).collect();
            let buf_b: Vec<f32> = (0..(w * h)).map(|i| -(i as f32) * 0.5).collect();
            let mut a = ExprArena::new();
            let ba = a.declare_buffer(crate::arena::BufferDecl {
                width: w as u32,
                height: h as u32,
            });
            let bb = a.declare_buffer(crate::arena::BufferDecl {
                width: w as u32,
                height: h as u32,
            });
            let x = a.push_var(0);
            let y = a.push_var(1);
            let ga = a.push_gather(ba, x, y);
            let gb = a.push_gather(bb, y, x);
            let root = a.push_binary(OpKind::Add, ga, gb);
            let (xs, ys) = idx_lanes();
            check_against_interp(
                &a,
                root,
                &[buf_a.as_slice(), buf_b.as_slice()],
                xs,
                ys,
                "2-buf",
            );
        }

        #[test]
        fn matmul_reduce_jit_matches_interpreter() {
            // out(j) = Σ_i W(i,j) * input(i), evaluated per output lane j = X.
            // The reduction over i unrolls to a flat gather/FMA chain (bound
            // extent), and the whole thing runs as one CtxKernelFn.
            //   W is IN×OUT row-major (width=IN, height=OUT); input is length IN.
            let (in_dim, out_dim) = (4usize, 6usize);
            let w: Vec<f32> = (0..(in_dim * out_dim))
                .map(|k| (k as f32) * 0.5 - 2.0)
                .collect();
            let input: Vec<f32> = (0..in_dim).map(|k| k as f32 + 1.0).collect();

            let mut a = ExprArena::new();
            let wb = a.declare_buffer(crate::arena::BufferDecl {
                width: in_dim as u32,
                height: out_dim as u32,
            });
            let ib = a.declare_buffer(crate::arena::BufferDecl {
                width: in_dim as u32,
                height: 1,
            });
            // body(i, j=X) = W(i, X) * input(i, 0)
            let i = a.push_var(4);
            let j = a.push_var(0);
            let zero = a.push_const(0.0);
            let wg = a.push_gather(wb, i, j);
            let ig = a.push_gather(ib, i, zero);
            let prod = a.push_binary(OpKind::Mul, wg, ig);
            let root = a.push_reduce(OpKind::Add, 4, in_dim as u32, prod);

            let buffers: &[&[f32]] = &[w.as_slice(), input.as_slice()];
            // Output lanes j = 0..6 (rest clamp to the last row, harmless here).
            let xs = [
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0,
            ];
            let ys = [0.0f32; 16];
            check_against_interp(&a, root, buffers, xs, ys, "matmul");
        }

        /// sqrt(X*X + Y*Y) - Z, with a non-commutative shape and FMA-able terms,
        /// fitting in registers (no spill).
        #[test]
        fn avx512_arith_no_spill() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let z = a.push_var(2);
            let xx = a.push_binary(OpKind::Mul, x, x);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let sum = a.push_binary(OpKind::Add, xx, yy);
            let dist = a.push_unary(OpKind::Sqrt, sum);
            let root = a.push_binary(OpKind::Sub, dist, z);

            let res = compile_arena_dag_avx512(&a, root).expect("avx512 compile");
            assert_eq!(res.spill_count, 0, "should fit without spilling");

            let (xs, ys, zs) = lanes();
            check(
                run16(&res, xs, ys, zs),
                |i| (xs[i] * xs[i] + ys[i] * ys[i]).sqrt() - zs[i],
                "norm-z",
            );
        }

        /// A wide expression that exceeds the 6 allocatable zmm regs, forcing a
        /// real 64-byte-slot stack frame (the SSE2 red zone cannot hold a zmm).
        #[test]
        fn avx512_spills_to_real_frame() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let mut terms = alloc::vec::Vec::new();
            for i in 1..=10u32 {
                let c = a.push_const(i as f32);
                let ax = a.push_binary(OpKind::Add, x, c);
                let by = a.push_binary(OpKind::Add, y, c);
                terms.push(a.push_binary(OpKind::Mul, ax, by));
            }
            while terms.len() > 1 {
                let mut next = alloc::vec::Vec::new();
                for pair in terms.chunks(2) {
                    if pair.len() == 2 {
                        next.push(a.push_binary(OpKind::Add, pair[0], pair[1]));
                    } else {
                        next.push(pair[0]);
                    }
                }
                terms = next;
            }
            let root = terms[0];

            let res = compile_arena_dag_avx512(&a, root).expect("avx512 compile");
            assert!(res.spill_count > 0, "expected spilling");

            let (xs, ys, zs) = lanes();
            check(
                run16(&res, xs, ys, zs),
                |i| {
                    let mut acc = 0.0f32;
                    for k in 1..=10u32 {
                        acc += (xs[i] + k as f32) * (ys[i] + k as f32);
                    }
                    acc
                },
                "spill",
            );
        }

        /// Compare + select with non-exclusive arms: `(X < Y) ? X : Y` (== min).
        /// No guard region forms, so this is the plain vcmpps->vpmovm2d mask +
        /// vpternlogd blend path.
        #[test]
        fn avx512_compare_select_blend() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let cond = a.push_binary(OpKind::Lt, x, y);
            let root = a.push_ternary(OpKind::Select, cond, x, y);

            let res = compile_arena_dag_avx512(&a, root).expect("avx512 compile");
            let (xs, ys, zs) = lanes();
            check(run16(&res, xs, ys, zs), |i| xs[i].min(ys[i]), "lt-select");
        }

        /// Select with arm-exclusive subexpressions: `(X > 0) ? Y*Y*Y : Z+Z+Z`.
        /// Forms guard regions, exercising the vptestmd+kortestw short-circuit
        /// branches (all-false skips Y^3, all-true skips 3Z) plus the per-lane
        /// blend on mixed input.
        #[test]
        fn avx512_select_guards() {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let z = a.push_var(2);
            let zero = a.push_const(0.0);
            let cond = a.push_binary(OpKind::Gt, x, zero);
            let yy = a.push_binary(OpKind::Mul, y, y);
            let yyy = a.push_binary(OpKind::Mul, yy, y);
            let zz = a.push_binary(OpKind::Add, z, z);
            let zzz = a.push_binary(OpKind::Add, zz, z);
            let root = a.push_ternary(OpKind::Select, cond, yyy, zzz);

            let res = compile_arena_dag_avx512(&a, root).expect("avx512 compile");

            let allpos = [2.0f32; 16];
            let allneg = [-2.0f32; 16];
            let ys = core::array::from_fn::<f32, 16, _>(|i| i as f32 * 0.5 + 1.0);
            let zs = core::array::from_fn::<f32, 16, _>(|i| 3.0 - i as f32 * 0.25);
            check(
                run16(&res, allpos, ys, zs),
                |i| ys[i] * ys[i] * ys[i],
                "guard-true",
            );
            check(run16(&res, allneg, ys, zs), |i| 3.0 * zs[i], "guard-false");

            let mixed = core::array::from_fn::<f32, 16, _>(|i| if i % 2 == 0 { 1.0 } else { -1.0 });
            check(
                run16(&res, mixed, ys, zs),
                |i| {
                    if mixed[i] > 0.0 {
                        ys[i] * ys[i] * ys[i]
                    } else {
                        3.0 * zs[i]
                    }
                },
                "guard-mixed",
            );
        }

        /// Rounding via vrndscaleps (floor/ceil/round), each a single EVEX op.
        #[test]
        fn avx512_rounding() {
            // Mixed fractional/sign inputs so each rounding mode is distinct.
            let xs = core::array::from_fn::<f32, 16, _>(|i| (i as f32 - 8.0) * 0.7);
            let ones = [1.0f32; 16];
            for (op, f, tag) in [
                (OpKind::Floor, f32::floor as fn(f32) -> f32, "floor"),
                (OpKind::Ceil, f32::ceil as fn(f32) -> f32, "ceil"),
                (
                    OpKind::Round,
                    f32::round_ties_even as fn(f32) -> f32,
                    "round",
                ),
            ] {
                let mut a = ExprArena::new();
                let x = a.push_var(0);
                let root = a.push_unary(op, x);
                let res = compile_arena_dag_avx512(&a, root).expect("avx512 compile");
                check(run16(&res, xs, ones, ones), |i| f(xs[i]), tag);
            }
        }
    }

    // =========================================================================
    // Backend op-coverage completeness (docs/designs/2026-07-25-two-level-ir-
    // and-backend-completeness.md)
    // =========================================================================
    //
    // Turns "backend X silently doesn't support op Y" into a named, itemized
    // test failure instead of a gap nobody notices until something happens to
    // exercise it (this is exactly how AVX-512's binary-op dispatch sat at
    // 6-of-15 required ops — nothing enumerated "the ops every backend must
    // support" anywhere, so the hole was invisible until 36 tests failed by
    // accident the first time someone compiled with `-C target-feature=
    // +avx512f`).
    //
    // Each test below is scoped to a backend this build ACTUALLY compiles —
    // `x86_backend_covers_required_ops` always runs on x86-64,
    // `aarch64_backend_covers_required_ops` always runs on aarch64, and
    // `avx512_backend_covers_required_ops` only compiles (and only needs to
    // pass) when built with `avx512f` — the same feature gate
    // `compile_arena_dag_with_ctx` uses to select `Avx512Backend` in
    // production. On a default `cargo test --workspace` (no RUSTFLAGS) on
    // this x86-64 host, that means: the SSE2 test runs and must be green
    // (it is: X86Backend already covers every required op), and the AVX-512
    // test does not even compile — it isn't lying about passing, it simply
    // isn't part of this build. The moment someone builds with
    // `+avx512f` (exactly the multi-ISA completion work tracked separately),
    // this same test starts running and will fail loudly, by name, for every
    // op `avx512::emit_unary`/`emit_binary`/`emit_plan` doesn't yet cover —
    // rather than waiting for an unrelated test to trip over the gap.
    mod backend_op_coverage {
        use super::super::coverage::*;
        use super::*;

        /// Run one `ResolvedOp` through a backend and report whether it
        /// emitted without error. Backends disagree on failure signaling
        /// today (avx512 returns `Err`; x86-64/aarch64's leaf encoders
        /// `panic!` on an unhandled op) — `catch_unwind` treats both as the
        /// same "not supported" signal so this test doesn't have to care
        /// which convention a given backend uses.
        fn try_emit<B: IsaBackend>(backend: &mut B, op: ResolvedOp) -> bool {
            let plan = InstructionPlan {
                reloads: alloc::vec::Vec::new(),
                op,
                setup_mov: None,
                store: None,
            };
            let mut code = alloc::vec::Vec::new();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                backend.emit_plan(&mut code, &plan)
            }))
            .map(|r| r.is_ok())
            .unwrap_or(false)
        }

        /// Sweep the required unary/binary/shift op lists plus the two
        /// bespoke ternary shapes (`MulAdd`, `Select`) against `backend`,
        /// collecting every failure instead of stopping at the first one —
        /// a completeness gap is much cheaper to fix as an itemized list
        /// than rediscovered one `cargo test` run per missing op.
        fn assert_covers_required_ops<B: IsaBackend>(backend_name: &str, backend: &mut B) {
            // The two explicit `try_emit` calls below for MulAdd/Select are
            // this constant, unrolled by hand (each needs its own
            // `ResolvedOp` shape, so they aren't worth a generic loop) — kept
            // in sync deliberately rather than by a shared loop.
            debug_assert_eq!(REQUIRED_TERNARY_OPS, &[OpKind::MulAdd, OpKind::Select]);
            let mut missing = alloc::vec::Vec::new();

            for &op in REQUIRED_UNARY_OPS {
                if !try_emit(
                    backend,
                    ResolvedOp::Unary {
                        op,
                        dst: Reg(4),
                        src: Reg(5),
                    },
                ) {
                    missing.push(alloc::format!("unary {:?}", op));
                }
            }
            for &op in REQUIRED_BINARY_OPS {
                if !try_emit(
                    backend,
                    ResolvedOp::Binary {
                        op,
                        dst: Reg(4),
                        left: Reg(4),
                        right: Reg(5),
                    },
                ) {
                    missing.push(alloc::format!("binary {:?}", op));
                }
            }
            for &op in REQUIRED_SHIFT_OPS {
                if !try_emit(
                    backend,
                    ResolvedOp::ShiftImm {
                        op,
                        dst: Reg(4),
                        src: Reg(4),
                        amount: 1,
                    },
                ) {
                    missing.push(alloc::format!("shift {:?}", op));
                }
            }
            if !try_emit(
                backend,
                ResolvedOp::FusedMulAdd {
                    dst: Reg(4),
                    a: Reg(5),
                    b: Reg(6),
                },
            ) {
                missing.push(alloc::string::String::from("ternary MulAdd"));
            }
            if !try_emit(
                backend,
                ResolvedOp::Select {
                    dst: Reg(4),
                    if_true: Reg(5),
                    if_false: Reg(6),
                },
            ) {
                missing.push(alloc::string::String::from("ternary Select"));
            }

            assert!(
                missing.is_empty(),
                "{backend_name} is missing required ops: {missing:?} (see \
                 pixelflow-ir/src/backend/emit/coverage.rs for the full \
                 completeness contract)"
            );
        }

        // Same gate as the struct itself: the SSE2 backend only exists on the
        // baseline build (the wide builds compile it out entirely), so this
        // sweep runs exactly when there is a backend to sweep. The default
        // build is the baseline, so CI's default job always runs it.
        #[cfg(all(
            target_arch = "x86_64",
            not(target_feature = "avx2"),
            not(target_feature = "avx512f")
        ))]
        #[test]
        fn x86_backend_covers_required_ops() {
            assert_covers_required_ops("X86Backend (SSE2)", &mut X86Backend::default());
        }

        // Gated on arch alone, not `target_feature = "avx2"`: the sweep only
        // *encodes* — bytes into a Vec, never executed — so it needs no AVX2
        // hardware or build flags. Running it on every x86-64 build means a
        // coverage gap in the AVX2 backend fails the default CI job, not just
        // the one ISA-matrix leg that selects it.
        #[cfg(target_arch = "x86_64")]
        #[test]
        fn avx2_backend_covers_required_ops() {
            assert_covers_required_ops("Avx2Backend", &mut Avx2Backend);
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        #[test]
        fn avx512_backend_covers_required_ops() {
            assert_covers_required_ops("Avx512Backend", &mut Avx512Backend);
        }

        #[cfg(target_arch = "aarch64")]
        #[test]
        fn aarch64_backend_covers_required_ops() {
            let mut backend = Aarch64Backend {
                pool: ConstPool::new(),
                adr_patch_pos: 0,
                max_regs: EmitCtx::default().max_regs,
            };
            assert_covers_required_ops("Aarch64Backend", &mut backend);
        }
    }
}
