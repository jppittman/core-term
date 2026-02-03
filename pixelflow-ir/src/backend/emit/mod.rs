//! JIT code emission via catamorphism.
//!
//! The key insight: register allocation emerges from tree structure.
//! Sethi-Ullman labeling computes minimum registers needed.
//! Register assignment is a FUNCTION of tree position (depth), not stateful allocation.
//!
//! ```text
//! emit : Expr × Depth → (Code, Reg)
//! ```
//!
//! No explicit alloc/free - the recursion depth IS the register.
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
pub mod x86_64;

use crate::kind::OpKind;

#[cfg(feature = "alloc")]
use crate::expr::Expr;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Physical register index.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
}
