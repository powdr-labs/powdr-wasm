//! This module is largely a copy of extensions/rv32im/circuit/src/mul/execution.rs.
//! The only differences are:
//! - OpenVM's `MultiplicationExecutor` is wrapped to allow for trait implementations.
//! - In `execute_e12_impl`, we retrieve and add the frame pointer.
//! - Immediate operands are supported.
//!
//! This file could be condensed a lot if more of the OpenVM code was public.

use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::{
    adapters::BaseAluAdapterExecutor,
    execution::{vm_read_multiple_ops, vm_write_multiple_ops},
    utils::sign_extend_u32,
};
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_derive::PreflightExecutor;
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    LocalOpcode,
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_IMM_AS, RV32_REGISTER_AS},
};
use openvm_rv32im_circuit::MultiplicationExecutor as MultiplicationExecutorInner;
use openvm_rv32im_transpiler::MulOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::imm_to_bytes;

/// Newtype wrapper to satisfy orphan rules for trait implementations.
#[derive(Clone, PreflightExecutor)]
pub struct MultiplicationExecutor<const NUM_LIMBS: usize, const NUM_REG_OPS: usize>(
    pub  MultiplicationExecutorInner<
        BaseAluAdapterExecutor<NUM_LIMBS, NUM_REG_OPS>,
        NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);

impl<const NUM_LIMBS: usize, const NUM_REG_OPS: usize>
    MultiplicationExecutor<NUM_LIMBS, NUM_REG_OPS>
{
    pub fn new(adapter: BaseAluAdapterExecutor<NUM_LIMBS, NUM_REG_OPS>, offset: usize) -> Self {
        Self(MultiplicationExecutorInner::new(adapter, offset))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct MulPreCompute {
    /// Second operand value (if immediate) or register index (if register)
    c: u32,
    /// Result register index
    a: u32,
    /// First operand register index
    b: u32,
    /// Whether the second operand is an immediate
    is_imm: bool,
}

impl<const NUM_LIMBS: usize, const NUM_REG_OPS: usize>
    MultiplicationExecutor<NUM_LIMBS, NUM_REG_OPS>
{
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut MulPreCompute,
    ) -> Result<(), StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;
        let e_u32 = e.as_canonical_u32();
        if (d.as_canonical_u32() != RV32_REGISTER_AS)
            || !(e_u32 == RV32_IMM_AS || e_u32 == RV32_REGISTER_AS)
        {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        debug_assert_eq!(
            MulOpcode::from_usize(opcode.local_opcode_idx(self.0.offset)),
            MulOpcode::MUL
        );
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = MulPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32(),
            b: b.as_canonical_u32(),
            is_imm,
        };
        Ok(())
    }
}

impl<F, const NUM_LIMBS: usize, const NUM_REG_OPS: usize> InterpreterExecutor<F>
    for MultiplicationExecutor<NUM_LIMBS, NUM_REG_OPS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<MulPreCompute>()
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
        let data: &mut MulPreCompute = data.borrow_mut();
        self.pre_compute_impl(pc, inst, data)?;

        if data.is_imm {
            Ok(execute_e1_handler::<_, _, true, NUM_LIMBS, NUM_REG_OPS>)
        } else {
            Ok(execute_e1_handler::<_, _, false, NUM_LIMBS, NUM_REG_OPS>)
        }
    }
}

impl<F, const NUM_LIMBS: usize, const NUM_REG_OPS: usize> InterpreterMeteredExecutor<F>
    for MultiplicationExecutor<NUM_LIMBS, NUM_REG_OPS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<MulPreCompute>>()
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
        let data: &mut E2PreCompute<MulPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(pc, inst, &mut data.data)?;

        if data.data.is_imm {
            Ok(execute_e2_handler::<_, _, true, NUM_LIMBS, NUM_REG_OPS>)
        } else {
            Ok(execute_e2_handler::<_, _, false, NUM_LIMBS, NUM_REG_OPS>)
        }
    }
}

/// Wrapping multiplication of two byte arrays, returning the lower NUM_LIMBS bytes.
#[inline(always)]
fn wrapping_mul_bytes<const NUM_LIMBS: usize>(
    a: &[u8; NUM_LIMBS],
    b: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    let mut result = [0u8; NUM_LIMBS];
    let mut carry = 0u32;
    for i in 0..NUM_LIMBS {
        let mut sum = carry;
        for j in 0..=i {
            sum += (a[j] as u32) * (b[i - j] as u32);
        }
        result[i] = (sum & 0xFF) as u8;
        carry = sum >> 8;
    }
    result
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const NUM_LIMBS: usize,
    const NUM_REG_OPS: usize,
>(
    pre_compute: &MulPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
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

    let result = wrapping_mul_bytes(&rs1, &rs2);

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
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const NUM_LIMBS: usize,
    const NUM_REG_OPS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &MulPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<MulPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, E_IS_IMM, NUM_LIMBS, NUM_REG_OPS>(pre_compute, exec_state);
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const E_IS_IMM: bool,
    const NUM_LIMBS: usize,
    const NUM_REG_OPS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<MulPreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<MulPreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, E_IS_IMM, NUM_LIMBS, NUM_REG_OPS>(&pre_compute.data, exec_state);
    }
}
