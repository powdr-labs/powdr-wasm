//! This module is largely a copy of extensions/rv32im/circuit/src/divrem/execution.rs.
//! The only differences are:
//! - OpenVM's `DivRemExecutor` is wrapped to allow for trait implementations.
//! - In `execute_e12_impl`, we retrieve and add the frame pointer.
//! - Operations are generic over NUM_LIMBS to support both 32-bit and 64-bit.
//!
//! This file could be condensed a lot if more of the OpenVM code was public.

use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::adapters::{BaseAluAdapterExecutor, RV32_REGISTER_NUM_LIMBS, imm_to_bytes};
use crate::execution::{vm_read_multiple_ops, vm_write_multiple_ops};
use crate::utils::sign_extend_u32;
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_derive::PreflightExecutor;
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    LocalOpcode,
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_IMM_AS, RV32_REGISTER_AS},
};
use openvm_rv32im_circuit::DivRemExecutor as DivRemExecutorInner;
use openvm_rv32im_transpiler::DivRemOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

/// Newtype wrapper to satisfy orphan rules for trait implementations.
#[derive(Clone, PreflightExecutor)]
pub struct DivRemExecutor<const NUM_LIMBS: usize, const NUM_REG_OPS: usize>(
    pub  DivRemExecutorInner<
        BaseAluAdapterExecutor<NUM_LIMBS, NUM_REG_OPS>,
        NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);

impl<const NUM_LIMBS: usize, const NUM_REG_OPS: usize> DivRemExecutor<NUM_LIMBS, NUM_REG_OPS> {
    pub fn new(adapter: BaseAluAdapterExecutor<NUM_LIMBS, NUM_REG_OPS>, offset: usize) -> Self {
        Self(DivRemExecutorInner::new(adapter, offset))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct DivRemPreCompute {
    /// Second operand value (if immediate) or register index (if register)
    c: u32,
    /// Result register index
    a: u32,
    /// First operand register index
    b: u32,
    /// Whether the second operand is an immediate
    is_imm: bool,
}

impl<const NUM_LIMBS: usize, const NUM_REG_OPS: usize> DivRemExecutor<NUM_LIMBS, NUM_REG_OPS> {
    /// Return the local opcode.
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut DivRemPreCompute,
    ) -> Result<DivRemOpcode, StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let local_opcode = DivRemOpcode::from_usize(opcode.local_opcode_idx(self.0.offset));
        let e_u32 = e.as_canonical_u32();
        if (d.as_canonical_u32() != RV32_REGISTER_AS)
            || !(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = DivRemPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32(),
            b: b.as_canonical_u32(),
            is_imm,
        };
        Ok(local_opcode)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $local_opcode:ident, $num_limbs:expr, $num_reg_ops:expr) => {
        match ($is_imm, $local_opcode) {
            (true, DivRemOpcode::DIV) => {
                Ok($execute_impl::<_, _, true, DivOp, $num_limbs, $num_reg_ops>)
            }
            (true, DivRemOpcode::DIVU) => {
                Ok($execute_impl::<_, _, true, DivuOp, $num_limbs, $num_reg_ops>)
            }
            (true, DivRemOpcode::REM) => {
                Ok($execute_impl::<_, _, true, RemOp, $num_limbs, $num_reg_ops>)
            }
            (true, DivRemOpcode::REMU) => {
                Ok($execute_impl::<_, _, true, RemuOp, $num_limbs, $num_reg_ops>)
            }
            (false, DivRemOpcode::DIV) => {
                Ok($execute_impl::<_, _, false, DivOp, $num_limbs, $num_reg_ops>)
            }
            (false, DivRemOpcode::DIVU) => {
                Ok($execute_impl::<_, _, false, DivuOp, $num_limbs, $num_reg_ops>)
            }
            (false, DivRemOpcode::REM) => {
                Ok($execute_impl::<_, _, false, RemOp, $num_limbs, $num_reg_ops>)
            }
            (false, DivRemOpcode::REMU) => {
                Ok($execute_impl::<_, _, false, RemuOp, $num_limbs, $num_reg_ops>)
            }
        }
    };
}

impl<F, const NUM_LIMBS: usize, const NUM_REG_OPS: usize> InterpreterExecutor<F>
    for DivRemExecutor<NUM_LIMBS, NUM_REG_OPS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<DivRemPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut DivRemPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        let is_imm = data.is_imm;

        dispatch!(
            execute_e1_handler,
            is_imm,
            local_opcode,
            NUM_LIMBS,
            NUM_REG_OPS
        )
    }
}

impl<F, const NUM_LIMBS: usize, const NUM_REG_OPS: usize> InterpreterMeteredExecutor<F>
    for DivRemExecutor<NUM_LIMBS, NUM_REG_OPS>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<DivRemPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<DivRemPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        let is_imm = data.data.is_imm;
        dispatch!(
            execute_e2_handler,
            is_imm,
            local_opcode,
            NUM_LIMBS,
            NUM_REG_OPS
        )
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    OP: DivRemOp<NUM_LIMBS>,
    const NUM_LIMBS: usize,
    const NUM_REG_OPS: usize,
>(
    pre_compute: &DivRemPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    const { assert!(NUM_LIMBS == 4 || NUM_LIMBS == 8) };
    const { assert!(NUM_REG_OPS * RV32_REGISTER_NUM_LIMBS == NUM_LIMBS) };

    let fp = exec_state.fp();
    let rs1 = vm_read_multiple_ops::<NUM_LIMBS, NUM_REG_OPS, _, _>(
        exec_state,
        RV32_REGISTER_AS,
        fp + pre_compute.b,
    );
    let rs2 = if E_IS_IMM {
        sign_extend_u32::<NUM_LIMBS>(pre_compute.c)
    } else {
        vm_read_multiple_ops::<NUM_LIMBS, NUM_REG_OPS, _, _>(
            exec_state,
            RV32_REGISTER_AS,
            fp + pre_compute.c,
        )
    };
    let result = <OP as DivRemOp<_>>::compute(rs1, rs2);

    vm_write_multiple_ops::<NUM_LIMBS, NUM_REG_OPS, _, _>(
        exec_state,
        RV32_REGISTER_AS,
        fp + pre_compute.a,
        &result,
    );
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    OP: DivRemOp<NUM_LIMBS>,
    const NUM_LIMBS: usize,
    const NUM_REG_OPS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &DivRemPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<DivRemPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, E_IS_IMM, OP, NUM_LIMBS, NUM_REG_OPS>(pre_compute, exec_state);
    }
}

#[create_handler]
#[inline(always)]
fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const E_IS_IMM: bool,
    OP: DivRemOp<NUM_LIMBS>,
    const NUM_LIMBS: usize,
    const NUM_REG_OPS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<DivRemPreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<DivRemPreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, E_IS_IMM, OP, NUM_LIMBS, NUM_REG_OPS>(
            &pre_compute.data,
            exec_state,
        );
    }
}

trait DivRemOp<const NUM_LIMBS: usize> {
    fn compute(rs1: [u8; NUM_LIMBS], rs2: [u8; NUM_LIMBS]) -> [u8; NUM_LIMBS];
}
struct DivOp;
struct DivuOp;
struct RemOp;
struct RemuOp;

/// Sign-extend N bytes to i64. Only valid for N=4 or N=8.
#[inline(always)]
fn sign_extend<const N: usize>(bytes: &[u8; N]) -> i64 {
    const { assert!(N == 4 || N == 8) };
    if N == 4 {
        i32::from_le_bytes(bytes[..4].try_into().unwrap()) as i64
    } else {
        i64::from_le_bytes(bytes[..8].try_into().unwrap())
    }
}

/// Zero-extend N bytes to u64. Only valid for N=4 or N=8.
#[inline(always)]
fn zero_extend<const N: usize>(bytes: &[u8; N]) -> u64 {
    const { assert!(N == 4 || N == 8) };
    if N == 4 {
        u32::from_le_bytes(bytes[..4].try_into().unwrap()) as u64
    } else {
        u64::from_le_bytes(bytes[..8].try_into().unwrap())
    }
}

// Signed division: div-by-zero → all-ones, overflow (MIN / -1) → dividend.
impl<const N: usize> DivRemOp<N> for DivOp {
    #[inline(always)]
    fn compute(rs1: [u8; N], rs2: [u8; N]) -> [u8; N] {
        const { assert!(N == 4 || N == 8) };
        let rs1_val = sign_extend(&rs1);
        let rs2_val = sign_extend(&rs2);
        match (rs1_val, rs2_val) {
            (_, 0) => [u8::MAX; N],
            // For N=8, this handles the signed overflow case (i64::MIN / -1).
            // For N=4, sign-extending i32::MIN gives -2147483648_i64 (not i64::MIN),
            // so this arm is unreachable. The default arm computes
            // -2147483648_i64 / -1_i64 = 2147483648_i64, which truncates back to
            // i32::MIN bytes — the correct RISC-V result (dividend).
            (i64::MIN, -1) => rs1,
            _ => (rs1_val / rs2_val).to_le_bytes()[..N].try_into().unwrap(),
        }
    }
}

// Unsigned division: div-by-zero → all-ones
impl<const N: usize> DivRemOp<N> for DivuOp {
    #[inline(always)]
    fn compute(rs1: [u8; N], rs2: [u8; N]) -> [u8; N] {
        const { assert!(N == 4 || N == 8) };
        let rs1_val = zero_extend(&rs1);
        let rs2_val = zero_extend(&rs2);
        match rs2_val {
            0 => [u8::MAX; N],
            _ => (rs1_val / rs2_val).to_le_bytes()[..N].try_into().unwrap(),
        }
    }
}

// Signed remainder: div-by-zero → dividend, overflow (MIN % -1) → zero.
impl<const N: usize> DivRemOp<N> for RemOp {
    #[inline(always)]
    fn compute(rs1: [u8; N], rs2: [u8; N]) -> [u8; N] {
        const { assert!(N == 4 || N == 8) };
        let rs1_val = sign_extend(&rs1);
        let rs2_val = sign_extend(&rs2);
        match (rs1_val, rs2_val) {
            (_, 0) => rs1,
            // See DivOp comment: for N=4 this arm is unreachable.
            (i64::MIN, -1) => [0; N],
            _ => (rs1_val % rs2_val).to_le_bytes()[..N].try_into().unwrap(),
        }
    }
}

// Unsigned remainder: div-by-zero → dividend
impl<const N: usize> DivRemOp<N> for RemuOp {
    #[inline(always)]
    fn compute(rs1: [u8; N], rs2: [u8; N]) -> [u8; N] {
        const { assert!(N == 4 || N == 8) };
        let rs1_val = zero_extend(&rs1);
        let rs2_val = zero_extend(&rs2);
        match rs2_val {
            0 => rs1,
            _ => (rs1_val % rs2_val).to_le_bytes()[..N].try_into().unwrap(),
        }
    }
}
