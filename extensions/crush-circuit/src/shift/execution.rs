//! This module is largely a copy of extensions/rv32im/circuit/src/shift/execution.rs.
//! The only differences are:
//! - OpenVM's `ShiftExecutor` is wrapped to allow for trait implementations.
//! - In `execute_e12_impl`, we retrieve and add the frame pointer.
//!
//! This file could be condensed a lot if more of the OpenVM code was public.

use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::execution::{vm_read_multiple_ops, vm_write_multiple_ops};
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_derive::PreflightExecutor;
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_IMM_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_rv32im_circuit::ShiftExecutor as ShiftExecutorInner;
use openvm_rv32im_transpiler::ShiftOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::{imm_to_bytes, BaseAluAdapterExecutor, RV32_REGISTER_NUM_LIMBS};
use crate::utils::sign_extend_u32;

/// Newtype wrapper to satisfy orphan rules for trait implementations.
#[derive(Clone, PreflightExecutor)]
pub struct ShiftExecutor<const NUM_LIMBS: usize, const NUM_REG_OPS: usize>(
    pub  ShiftExecutorInner<
        BaseAluAdapterExecutor<NUM_LIMBS, NUM_REG_OPS>,
        NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);

impl<const NUM_LIMBS: usize, const NUM_REG_OPS: usize> ShiftExecutor<NUM_LIMBS, NUM_REG_OPS> {
    pub fn new(adapter: BaseAluAdapterExecutor<NUM_LIMBS, NUM_REG_OPS>, offset: usize) -> Self {
        Self(ShiftExecutorInner::new(adapter, offset))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct ShiftPreCompute {
    /// Second operand value (if immediate) or register index (if register)
    c: u32,
    /// Result register index
    a: u32,
    /// First operand register index
    b: u32,
}

impl<const NUM_LIMBS: usize, const NUM_REG_OPS: usize> ShiftExecutor<NUM_LIMBS, NUM_REG_OPS> {
    /// Return `(is_imm, opcode)`.
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut ShiftPreCompute,
    ) -> Result<(bool, ShiftOpcode), StaticProgramError> {
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
        // Shift64Opcode has the same variant order as ShiftOpcode
        let local_opcode = ShiftOpcode::from_usize(opcode.local_opcode_idx(self.0.offset));
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = ShiftPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32(),
            b: b.as_canonical_u32(),
        };
        Ok((is_imm, local_opcode))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $opcode:ident, $num_limbs:expr, $num_reg_ops:expr) => {
        match ($is_imm, $opcode) {
            (true, ShiftOpcode::SLL) => {
                Ok($execute_impl::<_, _, true, { ShiftOpcode::SLL as u8 }, $num_limbs, $num_reg_ops>)
            }
            (false, ShiftOpcode::SLL) => {
                Ok($execute_impl::<_, _, false, { ShiftOpcode::SLL as u8 }, $num_limbs, $num_reg_ops>)
            }
            (true, ShiftOpcode::SRL) => {
                Ok($execute_impl::<_, _, true, { ShiftOpcode::SRL as u8 }, $num_limbs, $num_reg_ops>)
            }
            (false, ShiftOpcode::SRL) => {
                Ok($execute_impl::<_, _, false, { ShiftOpcode::SRL as u8 }, $num_limbs, $num_reg_ops>)
            }
            (true, ShiftOpcode::SRA) => {
                Ok($execute_impl::<_, _, true, { ShiftOpcode::SRA as u8 }, $num_limbs, $num_reg_ops>)
            }
            (false, ShiftOpcode::SRA) => {
                Ok($execute_impl::<_, _, false, { ShiftOpcode::SRA as u8 }, $num_limbs, $num_reg_ops>)
            }
        }
    };
}

impl<F, const NUM_LIMBS: usize, const NUM_REG_OPS: usize> InterpreterExecutor<F>
    for ShiftExecutor<NUM_LIMBS, NUM_REG_OPS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<ShiftPreCompute>()
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
        let data: &mut ShiftPreCompute = data.borrow_mut();
        let (is_imm, opcode) = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, is_imm, opcode, NUM_LIMBS, NUM_REG_OPS)
    }
}

impl<F, const NUM_LIMBS: usize, const NUM_REG_OPS: usize> InterpreterMeteredExecutor<F>
    for ShiftExecutor<NUM_LIMBS, NUM_REG_OPS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<ShiftPreCompute>>()
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
        let data: &mut E2PreCompute<ShiftPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, opcode) = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, is_imm, opcode, NUM_LIMBS, NUM_REG_OPS)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const OPCODE: u8,
    const NUM_LIMBS: usize,
    const NUM_REG_OPS: usize,
>(
    pre_compute: &ShiftPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    const { assert!(NUM_LIMBS == 4 || NUM_LIMBS == 8) };
    const { assert!(NUM_REG_OPS * RV32_REGISTER_NUM_LIMBS == NUM_LIMBS) };

    let num_bits = NUM_LIMBS * 8;
    let fp = exec_state.extra_regs()[0];
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

    // Shift amount is taken mod num_bits, from the lowest byte only
    let shift_amount = (rs2[0] as u32) % (num_bits as u32);

    // Convert byte arrays to native u64 for computation
    let val = u64::from_le_bytes(std::array::from_fn(
        |i| if i < NUM_LIMBS { rs1[i] } else { 0 },
    ));

    let result = match OPCODE {
        x if x == ShiftOpcode::SLL as u8 => val << shift_amount,
        x if x == ShiftOpcode::SRL as u8 => val >> shift_amount,
        x if x == ShiftOpcode::SRA as u8 => {
            // Sign-extend to i64 based on the actual bit width
            let signed = match NUM_LIMBS {
                4 => (val as u32 as i32 as i64) >> shift_amount,
                8 => (val as i64) >> shift_amount,
                _ => unreachable!(),
            };
            signed as u64
        }
        _ => unreachable!(),
    };

    let result_bytes = result.to_le_bytes();
    let result_val: [u8; NUM_LIMBS] = std::array::from_fn(|i| result_bytes[i]);

    vm_write_multiple_ops::<NUM_LIMBS, NUM_REG_OPS, _, _>(
        exec_state,
        RV32_REGISTER_AS,
        fp + pre_compute.a,
        &result_val,
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
    const OPCODE: u8,
    const NUM_LIMBS: usize,
    const NUM_REG_OPS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &ShiftPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<ShiftPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, E_IS_IMM, OPCODE, NUM_LIMBS, NUM_REG_OPS>(
            pre_compute,
            exec_state,
        );
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const E_IS_IMM: bool,
    const OPCODE: u8,
    const NUM_LIMBS: usize,
    const NUM_REG_OPS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<ShiftPreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<ShiftPreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, E_IS_IMM, OPCODE, NUM_LIMBS, NUM_REG_OPS>(
            &pre_compute.data,
            exec_state,
        );
    }
}
