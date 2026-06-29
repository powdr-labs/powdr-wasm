//! This module is largely a copy of extensions/rv32im/circuit/src/less_than/execution.rs.
//! The only differences are:
//! - OpenVM's `LessThanExecutor` is wrapped to allow for trait implementations.
//! - In `execute_e12_impl`, we retrieve and add the frame pointer.
//!
//! This file could be condensed a lot if more of the OpenVM code was public.

use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::execution::vm_read_multiple_ops;
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_derive::PreflightExecutor;
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    LocalOpcode,
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_IMM_AS, RV32_REGISTER_AS},
};
use openvm_rv32im_circuit::LessThanExecutor as LessThanExecutorInner;
use openvm_rv32im_transpiler::LessThanOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::{
    BaseAluAdapterExecutorDifferentInputsOutputs, RV32_REGISTER_NUM_LIMBS, imm_to_bytes,
};
use crate::utils::to_u64;

/// Newtype wrapper to satisfy orphan rules for trait implementations.
#[derive(Clone, PreflightExecutor)]
pub struct LessThanExecutor<
    const NUM_LIMBS: usize,
    const NUM_READ_OPS: usize,
    const NUM_WRITE_OPS: usize,
>(
    pub  LessThanExecutorInner<
        BaseAluAdapterExecutorDifferentInputsOutputs<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS>,
        NUM_LIMBS,
        RV32_CELL_BITS,
    >,
);

impl<const NUM_LIMBS: usize, const NUM_READ_OPS: usize, const NUM_WRITE_OPS: usize>
    LessThanExecutor<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS>
{
    pub fn new(
        adapter: BaseAluAdapterExecutorDifferentInputsOutputs<
            NUM_LIMBS,
            NUM_READ_OPS,
            NUM_WRITE_OPS,
        >,
        offset: usize,
    ) -> Self {
        Self(LessThanExecutorInner::new(adapter, offset))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct LessThanPreCompute {
    /// Second operand value (if immediate) or register index (if register)
    c: u32,
    /// Result register index
    a: u32,
    /// First operand register index
    b: u32,
}

impl<const NUM_LIMBS: usize, const NUM_READ_OPS: usize, const NUM_WRITE_OPS: usize>
    LessThanExecutor<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS>
{
    /// Return `(is_imm, is_sltu)`.
    #[inline(always)]
    pub(super) fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut LessThanPreCompute,
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
        // LessThan64Opcode has the same variant order as LessThanOpcode
        let local_opcode = LessThanOpcode::from_usize(opcode.local_opcode_idx(self.0.offset));
        let is_imm = e_u32 == RV32_IMM_AS;
        let c_u32 = c.as_canonical_u32();
        *data = LessThanPreCompute {
            c: if is_imm {
                u32::from_le_bytes(imm_to_bytes(c_u32))
            } else {
                c_u32
            },
            a: a.as_canonical_u32(),
            b: b.as_canonical_u32(),
        };
        Ok((is_imm, local_opcode == LessThanOpcode::SLTU))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_imm:ident, $is_sltu:ident, $num_limbs:expr, $num_read_ops:expr) => {
        match ($is_imm, $is_sltu) {
            (true, true) => Ok($execute_impl::<_, _, true, true, $num_limbs, $num_read_ops>),
            (true, false) => Ok($execute_impl::<_, _, true, false, $num_limbs, $num_read_ops>),
            (false, true) => Ok($execute_impl::<_, _, false, true, $num_limbs, $num_read_ops>),
            (false, false) => Ok($execute_impl::<_, _, false, false, $num_limbs, $num_read_ops>),
        }
    };
}

impl<F, const NUM_LIMBS: usize, const NUM_READ_OPS: usize, const NUM_WRITE_OPS: usize>
    InterpreterExecutor<F> for LessThanExecutor<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<LessThanPreCompute>()
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
        let data: &mut LessThanPreCompute = data.borrow_mut();
        let (is_imm, is_sltu) = self.pre_compute_impl(pc, inst, data)?;

        dispatch!(execute_e1_handler, is_imm, is_sltu, NUM_LIMBS, NUM_READ_OPS)
    }
}

impl<F, const NUM_LIMBS: usize, const NUM_READ_OPS: usize, const NUM_WRITE_OPS: usize>
    InterpreterMeteredExecutor<F> for LessThanExecutor<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<LessThanPreCompute>>()
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
        let data: &mut E2PreCompute<LessThanPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let (is_imm, is_sltu) = self.pre_compute_impl(pc, inst, &mut data.data)?;

        dispatch!(execute_e2_handler, is_imm, is_sltu, NUM_LIMBS, NUM_READ_OPS)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const E_IS_IMM: bool,
    const IS_U32: bool,
    const NUM_LIMBS: usize,
    const NUM_READ_OPS: usize,
>(
    pre_compute: &LessThanPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    const { assert!(NUM_LIMBS == 4 || NUM_LIMBS == 8) };
    const { assert!(NUM_READ_OPS * RV32_REGISTER_NUM_LIMBS == NUM_LIMBS) };

    let fp = exec_state.fp();
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
    let cmp_result = if IS_U32 {
        a < b
    } else if NUM_LIMBS == 4 {
        (a as i32) < (b as i32)
    } else {
        (a as i64) < (b as i64)
    };
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
    const IS_U32: bool,
    const NUM_LIMBS: usize,
    const NUM_READ_OPS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &LessThanPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<LessThanPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, E_IS_IMM, IS_U32, NUM_LIMBS, NUM_READ_OPS>(
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
    const IS_U32: bool,
    const NUM_LIMBS: usize,
    const NUM_READ_OPS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<LessThanPreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<LessThanPreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, E_IS_IMM, IS_U32, NUM_LIMBS, NUM_READ_OPS>(
            &pre_compute.data,
            exec_state,
        );
    }
}
