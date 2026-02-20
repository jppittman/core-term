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
//! ## Graph Coloring (for DAGs)
//!
//! For expressions with shared subexpressions (from e-graph extraction),
//! graph coloring handles liveness properly. See [`regalloc`] module.
//!
//! ## Spilling
//!
//! When `max_regs` is set lower than needed, we spill to stack:
//! - Spilled values stored via STR to [SP, #offset]
//! - Reloaded via LDR to dedicated reload register before use
//! - This lets ML models learn register pressure vs spill tradeoffs

pub mod aarch64;
pub mod executable;
pub mod lower;
#[cfg(feature = "alloc")]
pub mod regalloc;
pub mod x86_64;

use crate::kind::OpKind;

#[cfg(feature = "alloc")]
use crate::expr::Expr;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Physical register index.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Reg(pub u8);

/// Location of a value: either in a register or spilled to stack.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Loc {
    /// Value is in a register.
    Reg(Reg),
    /// Value is spilled to stack at this offset from SP.
    Spill(u16),
}

impl Loc {
    /// Get the register, panicking if spilled.
    pub fn reg(self) -> Reg {
        match self {
            Loc::Reg(r) => r,
            Loc::Spill(off) => panic!("expected register, got spill slot {}", off),
        }
    }
}

/// Emission context with register budget for ML training.
#[derive(Clone, Debug)]
pub struct EmitCtx {
    /// Maximum scratch registers before spilling (ML parameter).
    /// Default: 24 (v4-v27 on ARM64).
    pub max_regs: u8,
    /// Current spill offset from SP.
    pub spill_offset: u16,
    /// Number of spills performed (for cost modeling).
    pub spill_count: u16,
}

impl Default for EmitCtx {
    fn default() -> Self {
        Self {
            max_regs: 24, // v4-v27
            spill_offset: 0,
            spill_count: 0,
        }
    }
}

impl EmitCtx {
    /// Create context with custom register budget.
    pub fn with_max_regs(max_regs: u8) -> Self {
        Self {
            max_regs,
            ..Default::default()
        }
    }

    /// Allocate a spill slot, returns offset.
    pub fn alloc_spill(&mut self) -> u16 {
        let off = self.spill_offset;
        self.spill_offset += 16; // 128-bit vector
        self.spill_count += 1;
        off
    }
}

/// Input registers: X=v0, Y=v1, Z=v2, W=v3
pub const INPUT_REGS: [Reg; 4] = [Reg(0), Reg(1), Reg(2), Reg(3)];

/// First scratch register (after inputs)
pub const SCRATCH_BASE: u8 = 4;

/// Dedicated register for reloading spilled values (ARM64: v27)
#[cfg(target_arch = "aarch64")]
pub const RELOAD_REG: Reg = Reg(27);

/// Dedicated register for reloading spilled values (x86-64: xmm12)
#[cfg(target_arch = "x86_64")]
pub const RELOAD_REG: Reg = Reg(12);

/// Sethi-Ullman label: minimum registers needed to evaluate this subtree.
/// This is a catamorphism (fold) over the expression tree.
#[cfg(feature = "alloc")]
pub fn needs(expr: &Expr) -> usize {
    match expr {
        // Leaves need 1 register to hold their value
        Expr::Var(_) => 1,
        Expr::Const(_) => 1,

        // Unary: same as child (result overwrites input)
        Expr::Unary(_, child) => needs(child),

        // Binary: Sethi-Ullman magic
        Expr::Binary(_, left, right) => {
            let l = needs(left);
            let r = needs(right);
            if l == r {
                l + 1
            } else {
                l.max(r)
            }
        }

        // Ternary: need to hold all three, then combine
        Expr::Ternary(_, a, b, c) => {
            let na = needs(a);
            let nb = needs(b);
            let nc = needs(c);
            // Conservative: max + ties
            let max = na.max(nb).max(nc);
            if (na == nb) || (nb == nc) || (na == nc) {
                max + 1
            } else {
                max
            }
        }

        Expr::Nary(_, children) => {
            children.iter().map(needs).max().unwrap_or(0) + children.len() - 1
        }
    }
}

// =============================================================================
// Functional Emitter (ARM64)
// =============================================================================

/// Emit code for an expression. Returns (code, result_register).
///
/// `depth` determines the scratch register for this node's result.
/// Register assignment is deterministic from tree position.
#[cfg(all(feature = "alloc", target_arch = "aarch64"))]
pub fn emit(expr: &Expr, depth: u8) -> (Vec<u8>, Reg) {
    // Use default context (no spilling unless depth exceeds 24)
    let mut ctx = EmitCtx::default();
    let (code, loc) = emit_with_ctx(expr, depth, &mut ctx);
    (code, loc.reg())
}

/// Emit with explicit context for spill tracking.
/// Returns location (register or spill slot).
#[cfg(all(feature = "alloc", target_arch = "aarch64"))]
pub fn emit_with_ctx(expr: &Expr, depth: u8, ctx: &mut EmitCtx) -> (Vec<u8>, Loc) {
    use aarch64::*;

    // Would we exceed register budget?
    let effective_depth = if depth >= ctx.max_regs {
        ctx.max_regs - 1 // Cap at max, will spill
    } else {
        depth
    };

    match expr {
        // Variables are already in input registers - no code needed
        Expr::Var(i) => {
            if *i >= 4 {
                panic!("variable index {} out of range", i);
            }
            (vec![], Loc::Reg(INPUT_REGS[*i as usize]))
        }

        // Constants load into scratch register at current depth
        Expr::Const(val) => {
            let dst = Reg(SCRATCH_BASE + effective_depth);
            let mut code = Vec::new();
            let scratch = [Reg(28), Reg(29), Reg(30), Reg(31)]; // compound scratch
            emit_fmov_imm(&mut code, dst, *val, scratch);
            (code, Loc::Reg(dst))
        }

        // Unary: evaluate child, apply op, result in same register
        Expr::Unary(op, child) => {
            let (mut code, src_loc) = emit_with_ctx(child, effective_depth, ctx);

            // Materialize if spilled
            let src = materialize(&mut code, src_loc);

            // Result goes in scratch at depth (may or may not equal src)
            let dst = Reg(SCRATCH_BASE + effective_depth);
            let scratch = [Reg(28), Reg(29), Reg(30), Reg(31)];

            emit_unary(&mut code, *op, dst, src, scratch);
            (code, Loc::Reg(dst))
        }

        // Binary: Sethi-Ullman ordering with spill support
        Expr::Binary(op, left, right) => {
            let n_l = needs(left);
            let n_r = needs(right);
            let dst = Reg(SCRATCH_BASE + effective_depth);

            // Check if we need to spill
            let next_depth = effective_depth + 1;
            let need_spill = next_depth >= ctx.max_regs;

            if n_l >= n_r {
                // Left is heavier: evaluate left first
                let (mut code, l_loc) = emit_with_ctx(left, effective_depth, ctx);

                // If we'd overflow on right, spill left's result
                let l_loc = if need_spill {
                    spill(&mut code, l_loc, ctx)
                } else {
                    l_loc
                };

                let (r_code, r_loc) = emit_with_ctx(right, next_depth.min(ctx.max_regs - 1), ctx);
                code.extend(r_code);

                // Materialize both operands
                let l_reg = materialize(&mut code, l_loc);
                let r_reg = materialize(&mut code, r_loc);

                emit_binary(&mut code, *op, dst, l_reg, r_reg);
                (code, Loc::Reg(dst))
            } else {
                // Right is heavier: evaluate right first
                let (mut code, r_loc) = emit_with_ctx(right, effective_depth, ctx);

                // If we'd overflow on left, spill right's result
                let r_loc = if need_spill {
                    spill(&mut code, r_loc, ctx)
                } else {
                    r_loc
                };

                let (l_code, l_loc) = emit_with_ctx(left, next_depth.min(ctx.max_regs - 1), ctx);
                code.extend(l_code);

                // Materialize both operands
                let l_reg = materialize(&mut code, l_loc);
                let r_reg = materialize(&mut code, r_loc);

                emit_binary(&mut code, *op, dst, l_reg, r_reg);
                (code, Loc::Reg(dst))
            }
        }

        // Ternary: c is accumulator for MulAdd
        Expr::Ternary(op, a, b, c) => {
            let dst = Reg(SCRATCH_BASE + effective_depth);

            match op {
                OpKind::MulAdd => {
                    // c flows to dst (accumulator), a and b are multiplied
                    // Evaluate all three, spilling as needed
                    let (mut code, c_loc) = emit_with_ctx(c, effective_depth, ctx);

                    // Spill c if needed before evaluating a
                    let need_spill_c = effective_depth + 1 >= ctx.max_regs;
                    let c_loc = if need_spill_c {
                        spill(&mut code, c_loc, ctx)
                    } else {
                        c_loc
                    };

                    let (a_code, a_loc) = emit_with_ctx(a, (effective_depth + 1).min(ctx.max_regs - 1), ctx);
                    code.extend(a_code);

                    // Spill a if needed before evaluating b
                    let need_spill_a = effective_depth + 2 >= ctx.max_regs;
                    let a_loc = if need_spill_a {
                        spill(&mut code, a_loc, ctx)
                    } else {
                        a_loc
                    };

                    let (b_code, b_loc) = emit_with_ctx(b, (effective_depth + 2).min(ctx.max_regs - 1), ctx);
                    code.extend(b_code);

                    // Materialize all operands
                    let c_reg = materialize(&mut code, c_loc);
                    let a_reg = materialize(&mut code, a_loc);
                    let b_reg = materialize(&mut code, b_loc);

                    // Copy c to dst if needed
                    if c_reg.0 != dst.0 {
                        emit_mov(&mut code, dst, c_reg);
                    }

                    // FMLA: dst += a * b
                    emit_fmla(&mut code, dst, a_reg, b_reg);
                    (code, Loc::Reg(dst))
                }

                OpKind::Select => {
                    // a is mask, b is if_true, c is if_false
                    let (mut code, a_loc) = emit_with_ctx(a, effective_depth, ctx);

                    let need_spill_a = effective_depth + 1 >= ctx.max_regs;
                    let a_loc = if need_spill_a {
                        spill(&mut code, a_loc, ctx)
                    } else {
                        a_loc
                    };

                    let (b_code, b_loc) = emit_with_ctx(b, (effective_depth + 1).min(ctx.max_regs - 1), ctx);
                    code.extend(b_code);

                    let need_spill_b = effective_depth + 2 >= ctx.max_regs;
                    let b_loc = if need_spill_b {
                        spill(&mut code, b_loc, ctx)
                    } else {
                        b_loc
                    };

                    let (c_code, c_loc) = emit_with_ctx(c, (effective_depth + 2).min(ctx.max_regs - 1), ctx);
                    code.extend(c_code);

                    let a_reg = materialize(&mut code, a_loc);
                    let b_reg = materialize(&mut code, b_loc);
                    let c_reg = materialize(&mut code, c_loc);

                    // BSL needs mask in dst
                    if a_reg.0 != dst.0 {
                        emit_mov(&mut code, dst, a_reg);
                    }

                    emit_bsl(&mut code, dst, b_reg, c_reg);
                    (code, Loc::Reg(dst))
                }

                OpKind::Clamp => {
                    // clamp(a, lo=b, hi=c) = max(min(a, c), b)
                    let (mut code, a_loc) = emit_with_ctx(a, effective_depth, ctx);

                    let need_spill_a = effective_depth + 1 >= ctx.max_regs;
                    let a_loc = if need_spill_a {
                        spill(&mut code, a_loc, ctx)
                    } else {
                        a_loc
                    };

                    let (b_code, b_loc) = emit_with_ctx(b, (effective_depth + 1).min(ctx.max_regs - 1), ctx);
                    code.extend(b_code);

                    let need_spill_b = effective_depth + 2 >= ctx.max_regs;
                    let b_loc = if need_spill_b {
                        spill(&mut code, b_loc, ctx)
                    } else {
                        b_loc
                    };

                    let (c_code, c_loc) = emit_with_ctx(c, (effective_depth + 2).min(ctx.max_regs - 1), ctx);
                    code.extend(c_code);

                    let a_reg = materialize(&mut code, a_loc);
                    let b_reg = materialize(&mut code, b_loc);
                    let c_reg = materialize(&mut code, c_loc);

                    emit_fmin(&mut code, dst, a_reg, c_reg);
                    emit_fmax(&mut code, dst, dst, b_reg);
                    (code, Loc::Reg(dst))
                }

                _ => panic!("ternary emit not implemented for {:?}", op),
            }
        }

        Expr::Nary(_, _) => panic!("Nary not supported in JIT"),
    }
}

/// Spill a location to stack if it's in a register.
/// Returns new location (always Spill).
#[cfg(all(feature = "alloc", target_arch = "aarch64"))]
fn spill(code: &mut Vec<u8>, loc: Loc, ctx: &mut EmitCtx) -> Loc {
    match loc {
        Loc::Reg(r) => {
            let offset = ctx.alloc_spill();
            aarch64::emit_str_sp(code, r, offset);
            Loc::Spill(offset)
        }
        Loc::Spill(_) => loc, // Already spilled
    }
}

/// Materialize a location to a register.
/// If spilled, loads to RELOAD_REG.
#[cfg(all(feature = "alloc", target_arch = "aarch64"))]
fn materialize(code: &mut Vec<u8>, loc: Loc) -> Reg {
    match loc {
        Loc::Reg(r) => r,
        Loc::Spill(offset) => {
            aarch64::emit_ldr_sp(code, RELOAD_REG, offset);
            RELOAD_REG
        }
    }
}

/// Emit MOV (vector register copy)
#[cfg(target_arch = "aarch64")]
fn emit_mov(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    // ORR Vd.16B, Vn.16B, Vn.16B
    let inst = 0x4EA01C00u32
        | (dst.0 as u32)
        | ((src.0 as u32) << 5)
        | ((src.0 as u32) << 16);
    code.extend_from_slice(&inst.to_le_bytes());
}

// =============================================================================
// Functional Emitter (x86-64)
// =============================================================================

#[cfg(all(feature = "alloc", target_arch = "x86_64"))]
pub fn emit(expr: &Expr, depth: u8) -> (Vec<u8>, Reg) {
    use x86_64::*;

    match expr {
        Expr::Var(i) => {
            if *i >= 4 {
                panic!("variable index {} out of range", i);
            }
            (vec![], INPUT_REGS[*i as usize])
        }

        Expr::Const(val) => {
            let dst = Reg(SCRATCH_BASE + depth);
            let mut code = Vec::new();
            let scratch = [Reg(13), Reg(14), Reg(15), Reg(15)];
            emit_const(&mut code, dst, *val, scratch);
            (code, dst)
        }

        Expr::Unary(op, child) => {
            let (mut code, src) = emit(child, depth);
            let dst = Reg(SCRATCH_BASE + depth);
            let scratch = [Reg(13), Reg(14), Reg(15), Reg(15)];
            emit_unary(&mut code, *op, dst, src, scratch);
            (code, dst)
        }

        Expr::Binary(op, left, right) => {
            let n_l = needs(left);
            let n_r = needs(right);
            let dst = Reg(SCRATCH_BASE + depth);

            if n_l >= n_r {
                let (mut code, l_reg) = emit(left, depth);
                let (r_code, r_reg) = emit(right, depth + 1);
                code.extend(r_code);
                emit_binary(&mut code, *op, dst, l_reg, r_reg);
                (code, dst)
            } else {
                let (mut code, r_reg) = emit(right, depth);
                let (l_code, l_reg) = emit(left, depth + 1);
                code.extend(l_code);
                emit_binary(&mut code, *op, dst, l_reg, r_reg);
                (code, dst)
            }
        }

        Expr::Ternary(op, a, b, c) => {
            let dst = Reg(SCRATCH_BASE + depth);

            match op {
                OpKind::MulAdd => {
                    // x86 doesn't have FMLA, use FMUL + FADD
                    let (mut code, a_reg) = emit(a, depth);
                    let (b_code, b_reg) = emit(b, depth + 1);
                    let (c_code, c_reg) = emit(c, depth + 2);

                    code.extend(b_code);
                    code.extend(c_code);

                    // dst = a * b
                    emit_binary(&mut code, OpKind::Mul, dst, a_reg, b_reg);
                    // dst = dst + c
                    emit_binary(&mut code, OpKind::Add, dst, dst, c_reg);
                    (code, dst)
                }

                _ => panic!("ternary emit not implemented for {:?}", op),
            }
        }

        Expr::Nary(_, _) => panic!("Nary not supported in JIT"),
    }
}

// =============================================================================
// High-level API
// =============================================================================

/// Compile result with metadata for ML training.
#[cfg(feature = "alloc")]
pub struct CompileResult {
    /// The executable code.
    pub code: executable::ExecutableCode,
    /// Number of spills performed.
    pub spill_count: u16,
    /// Total stack space used for spills (bytes).
    pub spill_bytes: u16,
    /// Register budget that was used.
    pub max_regs: u8,
}

/// Compile an expression to executable code.
#[cfg(all(feature = "alloc", target_arch = "aarch64"))]
pub fn compile(expr: &Expr) -> Result<executable::ExecutableCode, &'static str> {
    compile_with_ctx(expr, EmitCtx::default()).map(|r| r.code)
}

/// Compile a DAG (from e-graph extraction) using graph coloring.
///
/// Unlike `compile`, this handles shared subexpressions properly.
/// Each unique subexpression is evaluated exactly once and its result
/// is kept in a register (or spilled) for all uses.
#[cfg(all(feature = "alloc", target_arch = "aarch64"))]
pub fn compile_dag(expr: &Expr) -> Result<CompileResult, &'static str> {
    compile_dag_with_ctx(expr, EmitCtx::default())
}

/// Compile DAG with explicit register budget.
#[cfg(all(feature = "alloc", target_arch = "aarch64"))]
pub fn compile_dag_with_ctx(expr: &Expr, ctx: EmitCtx) -> Result<CompileResult, &'static str> {
    use regalloc::{ValueId, build_interference_graph, color_graph};
    use alloc::collections::BTreeMap;

    // Lower compound ops first
    let lowered = lower::lower(expr);

    // Step 1: Linearize DAG into schedule (topological order)
    // Each unique subexpression gets a ValueId
    let (schedule, _expr_to_value, uses_map) = linearize_dag(&lowered);

    // Step 2: Build interference graph from schedule
    let mut graph = build_interference_graph(&schedule, |v| {
        uses_map.get(&v).cloned().unwrap_or_default()
    });

    // Step 2.5: Precolor input variables to their fixed registers
    // Var(0) -> v0, Var(1) -> v1, etc.
    for (value_id, op) in &schedule {
        if let ScheduledOp::Var(i) = op {
            graph.precolor(*value_id, INPUT_REGS[*i as usize]);
        }
    }

    // Step 3: Color the graph
    let allocation = color_graph(&graph, ctx.max_regs, SCRATCH_BASE);

    // Step 4: Emit code using the register assignment
    let mut code = Vec::new();
    let mut spill_offset = 0u16;
    let mut spill_count = 0u16;

    // Build spill slots for spilled values
    let mut spill_slots: BTreeMap<ValueId, u16> = BTreeMap::new();
    for &v in &allocation.spilled {
        spill_slots.insert(v, spill_offset);
        spill_offset += 16; // 128-bit vector
        spill_count += 1;
    }

    // Prologue: allocate stack frame if we spilled
    let frame_size = (spill_offset + 15) & !15;
    if frame_size > 0 {
        aarch64::emit_sub_sp(&mut code, frame_size);
    }

    // Emit each scheduled operation
    for (value_id, op_info) in &schedule {
        let dst_loc = if let Some(&reg) = allocation.assignment.get(value_id) {
            Loc::Reg(reg)
        } else if let Some(&offset) = spill_slots.get(value_id) {
            // Compute to RELOAD_REG, then spill
            Loc::Spill(offset)
        } else {
            panic!("value {:?} has no assignment or spill slot", value_id);
        };

        emit_scheduled_op(&mut code, op_info, dst_loc, &allocation.assignment, &spill_slots)?;
    }

    // Get result register (the last value in schedule is the root)
    let root_value = schedule.last().map(|(v, _)| *v).expect("empty schedule");
    let result_reg = if let Some(&reg) = allocation.assignment.get(&root_value) {
        reg
    } else if let Some(&offset) = spill_slots.get(&root_value) {
        aarch64::emit_ldr_sp(&mut code, RELOAD_REG, offset);
        RELOAD_REG
    } else {
        panic!("root has no register");
    };

    // Move result to v0 if needed
    if result_reg.0 != 0 {
        emit_mov(&mut code, Reg(0), result_reg);
    }

    // Epilogue
    if frame_size > 0 {
        aarch64::emit_add_sp(&mut code, frame_size);
    }

    // RET
    code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

    let exec = unsafe { executable::ExecutableCode::from_code(&code)? };

    Ok(CompileResult {
        code: exec,
        spill_count,
        spill_bytes: spill_offset,
        max_regs: ctx.max_regs,
    })
}

/// Info about an operation in the schedule.
#[cfg(feature = "alloc")]
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
    Ternary(OpKind, regalloc::ValueId, regalloc::ValueId, regalloc::ValueId),
}

/// Linearize a DAG into a schedule with value IDs.
/// Returns (schedule, expr_ptr_to_value_id, uses_map)
#[cfg(feature = "alloc")]
fn linearize_dag(
    expr: &Expr,
) -> (
    Vec<(regalloc::ValueId, ScheduledOp)>,
    alloc::collections::BTreeMap<*const Expr, regalloc::ValueId>,
    alloc::collections::BTreeMap<regalloc::ValueId, Vec<regalloc::ValueId>>,
) {
    use alloc::collections::BTreeMap;
    use regalloc::ValueId;

    let mut schedule = Vec::new();
    let mut expr_to_value: BTreeMap<*const Expr, ValueId> = BTreeMap::new();
    let mut uses_map: BTreeMap<ValueId, Vec<ValueId>> = BTreeMap::new();
    let mut next_id = 0u32;

    fn visit(
        expr: &Expr,
        schedule: &mut Vec<(ValueId, ScheduledOp)>,
        expr_to_value: &mut BTreeMap<*const Expr, ValueId>,
        uses_map: &mut BTreeMap<ValueId, Vec<ValueId>>,
        next_id: &mut u32,
    ) -> ValueId {
        let ptr = expr as *const Expr;

        // Check if already scheduled (DAG sharing)
        if let Some(&id) = expr_to_value.get(&ptr) {
            return id;
        }

        let my_id = ValueId(*next_id);
        *next_id += 1;

        let op = match expr {
            Expr::Var(i) => ScheduledOp::Var(*i as u8),
            Expr::Const(v) => ScheduledOp::Const(*v),
            Expr::Unary(op, child) => {
                let child_id = visit(child, schedule, expr_to_value, uses_map, next_id);
                uses_map.entry(my_id).or_default().push(child_id);
                ScheduledOp::Unary(*op, child_id)
            }
            Expr::Binary(op, left, right) => {
                let l_id = visit(left, schedule, expr_to_value, uses_map, next_id);
                let r_id = visit(right, schedule, expr_to_value, uses_map, next_id);
                uses_map.entry(my_id).or_default().push(l_id);
                uses_map.entry(my_id).or_default().push(r_id);
                ScheduledOp::Binary(*op, l_id, r_id)
            }
            Expr::Ternary(op, a, b, c) => {
                let a_id = visit(a, schedule, expr_to_value, uses_map, next_id);
                let b_id = visit(b, schedule, expr_to_value, uses_map, next_id);
                let c_id = visit(c, schedule, expr_to_value, uses_map, next_id);
                uses_map.entry(my_id).or_default().push(a_id);
                uses_map.entry(my_id).or_default().push(b_id);
                uses_map.entry(my_id).or_default().push(c_id);
                ScheduledOp::Ternary(*op, a_id, b_id, c_id)
            }
            Expr::Nary(_, _) => panic!("Nary not supported in DAG compilation"),
        };

        schedule.push((my_id, op));
        expr_to_value.insert(ptr, my_id);
        my_id
    }

    visit(expr, &mut schedule, &mut expr_to_value, &mut uses_map, &mut next_id);

    (schedule, expr_to_value, uses_map)
}

/// Emit code for a scheduled operation.
#[cfg(all(feature = "alloc", target_arch = "aarch64"))]
fn emit_scheduled_op(
    code: &mut Vec<u8>,
    op: &ScheduledOp,
    dst_loc: Loc,
    assignment: &alloc::collections::BTreeMap<regalloc::ValueId, Reg>,
    spill_slots: &alloc::collections::BTreeMap<regalloc::ValueId, u16>,
) -> Result<(), &'static str> {
    use aarch64::*;

    // Helper to get a value into a register
    let get_reg = |code: &mut Vec<u8>, v: regalloc::ValueId| -> Reg {
        if let Some(&reg) = assignment.get(&v) {
            reg
        } else if let Some(&offset) = spill_slots.get(&v) {
            emit_ldr_sp(code, RELOAD_REG, offset);
            RELOAD_REG
        } else {
            panic!("value {:?} not found in assignment or spill slots", v);
        }
    };

    // Compute into a register (possibly RELOAD_REG if spilling)
    let compute_reg = match dst_loc {
        Loc::Reg(r) => r,
        Loc::Spill(_) => RELOAD_REG,
    };

    match op {
        ScheduledOp::Var(_) => {
            // Variables are precolored to input registers - no code needed.
            // The register allocator knows about this via precoloring.
            // The value is already in the correct register.
        }
        ScheduledOp::Const(val) => {
            let scratch = [Reg(28), Reg(29), Reg(30), Reg(31)];
            emit_fmov_imm(code, compute_reg, *val, scratch);
        }
        ScheduledOp::Unary(op_kind, child) => {
            let src = get_reg(code, *child);
            let scratch = [Reg(28), Reg(29), Reg(30), Reg(31)];
            emit_unary(code, *op_kind, compute_reg, src, scratch);
        }
        ScheduledOp::Binary(op_kind, left, right) => {
            let l_reg = get_reg(code, *left);
            let r_reg = get_reg(code, *right);
            emit_binary(code, *op_kind, compute_reg, l_reg, r_reg);
        }
        ScheduledOp::Ternary(op_kind, a, b, c) => {
            match op_kind {
                OpKind::MulAdd => {
                    let a_reg = get_reg(code, *a);
                    let b_reg = get_reg(code, *b);
                    let c_reg = get_reg(code, *c);

                    // Copy c to dst (accumulator)
                    if compute_reg.0 != c_reg.0 {
                        emit_mov_reg(code, compute_reg, c_reg);
                    }
                    emit_fmla(code, compute_reg, a_reg, b_reg);
                }
                OpKind::Select => {
                    let a_reg = get_reg(code, *a);
                    let b_reg = get_reg(code, *b);
                    let c_reg = get_reg(code, *c);

                    if compute_reg.0 != a_reg.0 {
                        emit_mov_reg(code, compute_reg, a_reg);
                    }
                    emit_bsl(code, compute_reg, b_reg, c_reg);
                }
                OpKind::Clamp => {
                    let a_reg = get_reg(code, *a);
                    let b_reg = get_reg(code, *b);
                    let c_reg = get_reg(code, *c);

                    emit_fmin(code, compute_reg, a_reg, c_reg);
                    emit_fmax(code, compute_reg, compute_reg, b_reg);
                }
                _ => return Err("unsupported ternary op in DAG compilation"),
            }
        }
    }

    // Spill if needed
    if let Loc::Spill(offset) = dst_loc {
        emit_str_sp(code, compute_reg, offset);
    }

    Ok(())
}

/// Emit MOV (vector register copy) - internal helper
#[cfg(target_arch = "aarch64")]
fn emit_mov_reg(code: &mut Vec<u8>, dst: Reg, src: Reg) {
    if dst.0 != src.0 {
        // ORR Vd.16B, Vn.16B, Vn.16B
        let inst = 0x4EA01C00u32
            | (dst.0 as u32)
            | ((src.0 as u32) << 5)
            | ((src.0 as u32) << 16);
        code.extend_from_slice(&inst.to_le_bytes());
    }
}

/// Compile with explicit register budget for ML training.
///
/// Returns compile result with spill statistics.
#[cfg(all(feature = "alloc", target_arch = "aarch64"))]
pub fn compile_with_ctx(expr: &Expr, mut ctx: EmitCtx) -> Result<CompileResult, &'static str> {
    // Lower compound ops to primitives
    let lowered = lower::lower(expr);

    // Emit body
    let (body_code, result_loc) = emit_with_ctx(&lowered, 0, &mut ctx);

    // Build final code with prologue/epilogue if we spilled
    let mut code = Vec::new();

    // Prologue: allocate stack frame if we spilled
    // Round up to 16-byte alignment (AArch64 SP must be 16-byte aligned)
    let frame_size = (ctx.spill_offset + 15) & !15;
    if frame_size > 0 {
        aarch64::emit_sub_sp(&mut code, frame_size);
    }

    // Body
    code.extend(body_code);

    // Move result to v0 if not already there
    let result_reg = match result_loc {
        Loc::Reg(r) => r,
        Loc::Spill(off) => {
            // Reload spilled result
            aarch64::emit_ldr_sp(&mut code, RELOAD_REG, off);
            RELOAD_REG
        }
    };

    if result_reg.0 != 0 {
        emit_mov(&mut code, Reg(0), result_reg);
    }

    // Epilogue: deallocate stack frame
    if frame_size > 0 {
        aarch64::emit_add_sp(&mut code, frame_size);
    }

    // RET
    code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

    let exec = unsafe { executable::ExecutableCode::from_code(&code)? };

    Ok(CompileResult {
        code: exec,
        spill_count: ctx.spill_count,
        spill_bytes: ctx.spill_offset,
        max_regs: ctx.max_regs,
    })
}

/// Compile an expression to executable code (x86-64).
#[cfg(all(feature = "alloc", target_arch = "x86_64"))]
pub fn compile(expr: &Expr) -> Result<executable::ExecutableCode, &'static str> {
    // Lower compound ops to primitives
    let lowered = lower::lower(expr);

    let (mut code, result_reg) = emit(&lowered, 0);

    // Move result to xmm0 if not already there
    if result_reg.0 != 0 {
        x86_64::emit_movaps(&mut code, Reg(0), result_reg);
    }

    // RET
    code.push(0xC3);

    unsafe { executable::ExecutableCode::from_code(&code) }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[cfg(feature = "alloc")]
mod tests {
    use super::*;

    #[test]
    fn test_needs_simple() {
        // X + Y: both leaves need 1, binary needs max(1,1)+1 = 2
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );
        assert_eq!(needs(&expr), 2);
    }

    #[test]
    fn test_needs_unbalanced() {
        // (X + Y) + Z: left needs 2, right needs 1, total = max(2,1) = 2
        let left = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );
        let expr = Expr::Binary(OpKind::Add, Box::new(left), Box::new(Expr::Var(2)));
        assert_eq!(needs(&expr), 2);
    }

    #[test]
    fn test_needs_balanced_deep() {
        // (X + Y) + (Z + W): both sides need 2, total = 2+1 = 3
        let left = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );
        let right = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(2)),
            Box::new(Expr::Var(3)),
        );
        let expr = Expr::Binary(OpKind::Add, Box::new(left), Box::new(right));
        assert_eq!(needs(&expr), 3);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_spill_forced() {
        // (X + Y) + (Z + W) needs 3 registers
        // With max_regs=2, we must spill
        let left = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );
        let right = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(2)),
            Box::new(Expr::Var(3)),
        );
        let expr = Expr::Binary(OpKind::Add, Box::new(left), Box::new(right));

        // Compile with only 2 scratch registers - should spill
        let ctx = EmitCtx::with_max_regs(2);
        let result = compile_with_ctx(&expr, ctx).expect("compile failed");

        // Should have spilled at least once
        assert!(result.spill_count > 0, "expected spills with max_regs=2");
        eprintln!("Spill count: {}, bytes: {}", result.spill_count, result.spill_bytes);

        // Verify it still computes correctly
        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(1.0);
            let y = vdupq_n_f32(2.0);
            let z = vdupq_n_f32(3.0);
            let w = vdupq_n_f32(4.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            // (1+2) + (3+4) = 3 + 7 = 10
            assert_eq!(vgetq_lane_f32(out, 0), 10.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_no_spill_with_enough_regs() {
        // Same expression but with enough registers - no spill
        let left = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );
        let right = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(2)),
            Box::new(Expr::Var(3)),
        );
        let expr = Expr::Binary(OpKind::Add, Box::new(left), Box::new(right));

        // Compile with default (24 registers) - no spill needed
        let ctx = EmitCtx::default();
        let result = compile_with_ctx(&expr, ctx).expect("compile failed");

        assert_eq!(result.spill_count, 0, "should not spill with 24 registers");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_spill_deeply_nested() {
        // Create a chain: ((((X + Y) + Z) + W) + X)
        // This naturally needs 2 registers, but with max_regs=1 we must spill
        let e1 = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );
        let e2 = Expr::Binary(OpKind::Add, Box::new(e1), Box::new(Expr::Var(2)));
        let e3 = Expr::Binary(OpKind::Add, Box::new(e2), Box::new(Expr::Var(3)));
        let expr = Expr::Binary(OpKind::Add, Box::new(e3), Box::new(Expr::Var(0)));

        let ctx = EmitCtx::with_max_regs(1);
        let result = compile_with_ctx(&expr, ctx).expect("compile failed");

        eprintln!("Deeply nested spill count: {}", result.spill_count);

        // Verify correctness
        unsafe {
            use core::arch::aarch64::*;
            let x = vdupq_n_f32(1.0);
            let y = vdupq_n_f32(2.0);
            let z = vdupq_n_f32(3.0);
            let w = vdupq_n_f32(4.0);

            let func: executable::KernelFn = result.code.as_fn();
            let out = func(x, y, z, w);
            // ((((1+2)+3)+4)+1) = 11
            assert_eq!(vgetq_lane_f32(out, 0), 11.0);
        }
    }

    // =========================================================================
    // Graph Coloring (DAG) Tests
    // =========================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dag_simple() {
        // Simple expression: X + Y (no sharing)
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );

        let result = compile_dag(&expr).expect("DAG compile failed");
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
    fn test_dag_with_constant() {
        // X * 2.0 + Y
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Binary(
                OpKind::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Var(1)),
        );

        let result = compile_dag(&expr).expect("DAG compile failed");

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
    fn test_dag_with_spill() {
        // Complex expression with limited registers
        let left = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );
        let right = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(2)),
            Box::new(Expr::Var(3)),
        );
        let expr = Expr::Binary(OpKind::Add, Box::new(left), Box::new(right));

        // Compile with only 2 registers - should require spilling
        let ctx = EmitCtx::with_max_regs(2);
        let result = compile_dag_with_ctx(&expr, ctx).expect("DAG compile failed");

        // Graph coloring may or may not spill depending on the graph structure
        // The important thing is correctness
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

    #[test]
    fn test_linearize_dag() {
        // Test the linearization function
        let expr = Expr::Binary(
            OpKind::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        );

        let (schedule, expr_to_value, uses_map) = linearize_dag(&expr);

        // Should have 3 values: X, Y, X+Y
        assert_eq!(schedule.len(), 3);

        // Root (X+Y) should use both X and Y
        let root_id = schedule.last().unwrap().0;
        let root_uses = uses_map.get(&root_id).expect("root should have uses");
        assert_eq!(root_uses.len(), 2);
    }
}
