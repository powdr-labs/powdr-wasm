//! This module is largely a copy of extensions/rv32im/circuit/src/base_alu/execution.rs.
//! The only differences are:
//! - OpenVM's `BaseAluExecutor` is wrapped to allow for trait implementations.
//! - In `execute_e12_impl`, we retrieve and add the frame pointer.
//!
//! This file could be condensed a lot if more of the OpenVM code was public.

use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::{
    execution::{vm_read_multiple_ops, vm_write_multiple_ops},
    utils::sign_extend_u32,
};
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_derive::PreflightExecutor;
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_IMM_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_circuit::BaseAluExecutor as BaseAluExecutorInner;
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::{imm_to_bytes, BaseAluAdapterExecutor};

/// Newtype wrapper to satisfy orphan rules for trait implementations.
#[derive(Clone, PreflightExecutor)]
pub struct BaseAluExecutor<const NUM_LIMBS: usize, const NUM_REG_OPS: usize>(
    pub  BaseAluExecutorInner<
        BaseAluAdapterExecutor<NUM_LIMBS, NUM_REG_OPS>,
        NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);

impl<const NUM_LIMBS: usize, const NUM_REG_OPS: usize> BaseAluExecutor<NUM_LIMBS, NUM_REG_OPS> {
    pub fn new(adapter: BaseAluAdapterExecutor<NUM_LIMBS, NUM_REG_OPS>, offset: usize) -> Self {
        Self(BaseAluExecutorInner::new(adapter, offset))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct BaseAluPreCompute {
    /// Second operand value (if immediate) or register index (if register)
    c: u32,
    /// Result register index
    a: u32,
    /// First operand register index
    b: u32,
}

impl<const NUM_LIMBS: usize, const NUM_REG_OPS: usize> BaseAluExecutor<NUM_LIMBS, NUM_REG_OPS> {
    /// Return `is_imm`, true if `e` is RV32_IMM_AS.
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut BaseAluPreCompute,
    ) -> Result<bool, StaticProgramError> {
        let Instruction { a, b, c, d, e, .. } = inst;
        let e_u32 = e.as_canonical_u32();
        if (d.as_canonical_u32() != RV32_REGISTER_AS)
            || !(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = BaseAluPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32(),
            b: b.as_canonical_u32(),
        };
        Ok(is_imm)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $opcode:expr, $offset:expr, $num_limbs:expr, $num_reg_ops:expr) => {
        Ok(
            // BaseAlu64Opcode has the same variant order as BaseAluOpcode
            match (
                $is_imm,
                BaseAluOpcode::from_usize($opcode.local_opcode_idx($offset)),
            ) {
                (true, BaseAluOpcode::ADD) => {
                    $execute_impl::<_, _, true, AddOp, $num_limbs, $num_reg_ops>
                }
                (false, BaseAluOpcode::ADD) => {
                    $execute_impl::<_, _, false, AddOp, $num_limbs, $num_reg_ops>
                }
                (true, BaseAluOpcode::SUB) => {
                    $execute_impl::<_, _, true, SubOp, $num_limbs, $num_reg_ops>
                }
                (false, BaseAluOpcode::SUB) => {
                    $execute_impl::<_, _, false, SubOp, $num_limbs, $num_reg_ops>
                }
                (true, BaseAluOpcode::XOR) => {
                    $execute_impl::<_, _, true, XorOp, $num_limbs, $num_reg_ops>
                }
                (false, BaseAluOpcode::XOR) => {
                    $execute_impl::<_, _, false, XorOp, $num_limbs, $num_reg_ops>
                }
                (true, BaseAluOpcode::OR) => {
                    $execute_impl::<_, _, true, OrOp, $num_limbs, $num_reg_ops>
                }
                (false, BaseAluOpcode::OR) => {
                    $execute_impl::<_, _, false, OrOp, $num_limbs, $num_reg_ops>
                }
                (true, BaseAluOpcode::AND) => {
                    $execute_impl::<_, _, true, AndOp, $num_limbs, $num_reg_ops>
                }
                (false, BaseAluOpcode::AND) => {
                    $execute_impl::<_, _, false, AndOp, $num_limbs, $num_reg_ops>
                }
            },
        )
    };
}

impl<F, const NUM_LIMBS: usize, const NUM_REG_OPS: usize> InterpreterExecutor<F>
    for BaseAluExecutor<NUM_LIMBS, NUM_REG_OPS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<BaseAluPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut BaseAluPreCompute = data.borrow_mut();
        let is_imm = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(
            execute_e1_handler,
            is_imm,
            inst.opcode,
            self.0.offset,
            NUM_LIMBS,
            NUM_REG_OPS
        )
    }
}

impl<F, const NUM_LIMBS: usize, const NUM_REG_OPS: usize> InterpreterMeteredExecutor<F>
    for BaseAluExecutor<NUM_LIMBS, NUM_REG_OPS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BaseAluPreCompute>>()
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
        let data: &mut E2PreCompute<BaseAluPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let is_imm = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(
            execute_e2_handler,
            is_imm,
            inst.opcode,
            self.0.offset,
            NUM_LIMBS,
            NUM_REG_OPS
        )
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_IMM: bool,
    OP: AluOp,
    const NUM_LIMBS: usize,
    const NUM_REG_OPS: usize,
>(
    pre_compute: &BaseAluPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    const { assert!(NUM_LIMBS == 4 || NUM_LIMBS == 8) };
    const { assert!(NUM_REG_OPS * RV32_REGISTER_NUM_LIMBS == NUM_LIMBS) };

    let fp = exec_state.extra_regs()[0];
    let rs1 = vm_read_multiple_ops::<NUM_LIMBS, NUM_REG_OPS, _, _>(
        exec_state,
        RV32_REGISTER_AS,
        fp + pre_compute.b,
    );
    let rs2 = if IS_IMM {
        sign_extend_u32::<NUM_LIMBS>(pre_compute.c)
    } else {
        vm_read_multiple_ops::<NUM_LIMBS, NUM_REG_OPS, _, _>(
            exec_state,
            RV32_REGISTER_AS,
            fp + pre_compute.c,
        )
    };
    let a = u64::from_le_bytes(std::array::from_fn(
        |i| if i < NUM_LIMBS { rs1[i] } else { 0 },
    ));
    let b = u64::from_le_bytes(std::array::from_fn(
        |i| if i < NUM_LIMBS { rs2[i] } else { 0 },
    ));
    let result = OP::compute(a, b).to_le_bytes();
    let rd: [u8; NUM_LIMBS] = std::array::from_fn(|i| result[i]);
    vm_write_multiple_ops::<NUM_LIMBS, NUM_REG_OPS, _, _>(
        exec_state,
        RV32_REGISTER_AS,
        fp + pre_compute.a,
        &rd,
    );
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_IMM: bool,
    OP: AluOp,
    const NUM_LIMBS: usize,
    const NUM_REG_OPS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &BaseAluPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<BaseAluPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, IS_IMM, OP, NUM_LIMBS, NUM_REG_OPS>(pre_compute, exec_state);
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_IMM: bool,
    OP: AluOp,
    const NUM_LIMBS: usize,
    const NUM_REG_OPS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<BaseAluPreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<BaseAluPreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, IS_IMM, OP, NUM_LIMBS, NUM_REG_OPS>(
            &pre_compute.data,
            exec_state,
        );
    }
}

// ==================== ALU operation trait ====================

trait AluOp {
    fn compute(a: u64, b: u64) -> u64;
}

struct AddOp;
struct SubOp;
struct XorOp;
struct OrOp;
struct AndOp;

impl AluOp for AddOp {
    #[inline(always)]
    fn compute(a: u64, b: u64) -> u64 {
        a.wrapping_add(b)
    }
}
impl AluOp for SubOp {
    #[inline(always)]
    fn compute(a: u64, b: u64) -> u64 {
        a.wrapping_sub(b)
    }
}
impl AluOp for XorOp {
    #[inline(always)]
    fn compute(a: u64, b: u64) -> u64 {
        a ^ b
    }
}
impl AluOp for OrOp {
    #[inline(always)]
    fn compute(a: u64, b: u64) -> u64 {
        a | b
    }
}
impl AluOp for AndOp {
    #[inline(always)]
    fn compute(a: u64, b: u64) -> u64 {
        a & b
    }
}
