//! Fast execution path for the EQ chip.
//!
//! This follows the same pattern as the LessThan execution module:
//! - `EqExecutor` wraps `EqExecutorInner` for orphan rule compliance
//! - `InterpreterExecutor` and `InterpreterMeteredExecutor` provide the fast path
//! - Frame pointer handling is the key difference from OpenVM's RISC-V chips

use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::{adapters::W32_REG_OPS, execution::vm_read_multiple_ops};
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_derive::PreflightExecutor;
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_crush_transpiler::EqOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;

use super::core::EqExecutorInner;
use crate::adapters::{
    imm_to_bytes, BaseAluAdapterExecutorDifferentInputsOutputs, RV32_REGISTER_NUM_LIMBS,
};
use crate::utils::to_u64;

/// Newtype wrapper to satisfy orphan rules for trait implementations.
#[derive(Clone, PreflightExecutor)]
pub struct EqExecutor<const NUM_LIMBS: usize, const NUM_READ_OPS: usize>(
    pub  EqExecutorInner<
        BaseAluAdapterExecutorDifferentInputsOutputs<NUM_LIMBS, NUM_READ_OPS, W32_REG_OPS>,
        NUM_LIMBS,
    >,
);

impl<const NUM_LIMBS: usize, const NUM_READ_OPS: usize> EqExecutor<NUM_LIMBS, NUM_READ_OPS> {
    pub fn new(
        adapter: BaseAluAdapterExecutorDifferentInputsOutputs<NUM_LIMBS, NUM_READ_OPS, W32_REG_OPS>,
        offset: usize,
    ) -> Self {
        Self(EqExecutorInner::new(adapter, offset))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct EqPreCompute {
    /// Second operand value (if immediate) or register index (if register)
    c: u32,
    /// Result register index
    a: u32,
    /// First operand register index
    b: u32,
}

impl<const NUM_LIMBS: usize, const NUM_READ_OPS: usize> EqExecutor<NUM_LIMBS, NUM_READ_OPS> {
    /// Return `(is_imm, is_neq)`.
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut EqPreCompute,
    ) -> Result<(bool, bool), StaticProgramError> {
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
        // Eq64Opcode has the same variant order as EqOpcode
        let local_opcode = EqOpcode::from_usize(opcode.local_opcode_idx(self.0.offset));
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = EqPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32(),
            b: b.as_canonical_u32(),
        };
        Ok((is_imm, local_opcode == EqOpcode::NEQ))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $is_neq:ident, $num_limbs:expr, $num_read_ops:expr) => {
        match ($is_imm, $is_neq) {
            (true, true) => Ok($execute_impl::<_, _, true, true, $num_limbs, $num_read_ops>),
            (true, false) => Ok($execute_impl::<_, _, true, false, $num_limbs, $num_read_ops>),
            (false, true) => Ok($execute_impl::<_, _, false, true, $num_limbs, $num_read_ops>),
            (false, false) => Ok($execute_impl::<_, _, false, false, $num_limbs, $num_read_ops>),
        }
    };
}

impl<F, const NUM_LIMBS: usize, const NUM_READ_OPS: usize> InterpreterExecutor<F>
    for EqExecutor<NUM_LIMBS, NUM_READ_OPS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<EqPreCompute>()
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
        let data: &mut EqPreCompute = data.borrow_mut();
        let (is_imm, is_neq) = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, is_imm, is_neq, NUM_LIMBS, NUM_READ_OPS)
    }
}

impl<F, const NUM_LIMBS: usize, const NUM_READ_OPS: usize> InterpreterMeteredExecutor<F>
    for EqExecutor<NUM_LIMBS, NUM_READ_OPS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<EqPreCompute>>()
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
        let data: &mut E2PreCompute<EqPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, is_neq) = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, is_imm, is_neq, NUM_LIMBS, NUM_READ_OPS)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const IS_NEQ: bool,
    const NUM_LIMBS: usize,
    const NUM_READ_OPS: usize,
>(
    pre_compute: &EqPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    const { assert!(NUM_LIMBS == 4 || NUM_LIMBS == 8) };
    const { assert!(NUM_READ_OPS * RV32_REGISTER_NUM_LIMBS == NUM_LIMBS) };

    let fp = exec_state.extra_regs()[0];
    let rs1 = vm_read_multiple_ops::<NUM_LIMBS, NUM_READ_OPS, _, _>(
        exec_state,
        RV32_REGISTER_AS,
        fp + pre_compute.b,
    );
    let a = to_u64(rs1);
    let b = if E_IS_IMM {
        // pre_compute.c is already sign-extended to 32 bits by pre_compute_impl
        if NUM_LIMBS == 4 {
            pre_compute.c as u64
        } else {
            pre_compute.c as i32 as i64 as u64
        }
    } else {
        let rs2 = vm_read_multiple_ops::<NUM_LIMBS, NUM_READ_OPS, _, _>(
            exec_state,
            RV32_REGISTER_AS,
            fp + pre_compute.c,
        );
        to_u64(rs2)
    };
    let cmp_result = if IS_NEQ { a != b } else { a == b };
    // Write only one register-width (4 bytes): comparison results are always i32,
    // even for 64-bit operands.
    let mut rd = [0u8; RV32_REGISTER_NUM_LIMBS];
    rd[0] = cmp_result as u8;
    exec_state.vm_write(RV32_REGISTER_AS, fp + pre_compute.a, &rd);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const IS_NEQ: bool,
    const NUM_LIMBS: usize,
    const NUM_READ_OPS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &EqPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<EqPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, E_IS_IMM, IS_NEQ, NUM_LIMBS, NUM_READ_OPS>(
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
    const IS_NEQ: bool,
    const NUM_LIMBS: usize,
    const NUM_READ_OPS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<EqPreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<EqPreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, E_IS_IMM, IS_NEQ, NUM_LIMBS, NUM_READ_OPS>(
            &pre_compute.data,
            exec_state,
        );
    }
}
