use std::{
    array,
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::*,
    system::memory::{POINTER_MAX_BITS, online::GuestMemory},
};
use openvm_circuit_derive::PreflightExecutor;
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    LocalOpcode,
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_IMM_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_rv32im_circuit::LoadSignExtendExecutor as LoadSignExtendExecutorInner;
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::LoadStoreAdapterExecutor;
use crate::memory_config::FpMemory;

/// Newtype wrapper to satisfy orphan rules for trait implementations.
#[derive(Clone, PreflightExecutor)]
pub struct LoadSignExtendExecutor<const NUM_LIMBS: usize>(
    pub LoadSignExtendExecutorInner<LoadStoreAdapterExecutor, NUM_LIMBS, RV32_CELL_BITS>,
);

impl<const NUM_LIMBS: usize> LoadSignExtendExecutor<NUM_LIMBS> {
    pub fn new(adapter: LoadStoreAdapterExecutor) -> Self {
        Self(LoadSignExtendExecutorInner::new(adapter))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct LoadSignExtendPreCompute {
    imm_extended: u32,
    a: u32,
    b: u32,
    e: u32,
}

impl LoadSignExtendExecutor<{ RV32_REGISTER_NUM_LIMBS }> {
    /// Return (is_loadb, enabled)
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut LoadSignExtendPreCompute,
    ) -> Result<(bool, bool), StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            ..
        } = inst;

        let e_u32 = e.as_canonical_u32();
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 == RV32_IMM_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode = Rv32LoadStoreOpcode::from_usize(
            opcode.local_opcode_idx(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );
        match local_opcode {
            LOADB | LOADH => {}
            _ => unreachable!("LoadSignExtendExecutor should only handle LOADB/LOADH opcodes"),
        }

        let imm_lo = c.as_canonical_u32();
        let imm_hi = g.as_canonical_u32();
        let imm_extended = imm_lo | (imm_hi << 16);

        *data = LoadSignExtendPreCompute {
            imm_extended,
            a: a.as_canonical_u32(),
            b: b.as_canonical_u32(),
            e: e_u32,
        };
        let enabled = !f.is_zero();
        Ok((local_opcode == LOADB, enabled))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $is_loadb:ident, $enabled:ident) => {
        match ($is_loadb, $enabled) {
            (true, true) => Ok($execute_impl::<_, _, true, true>),
            (true, false) => Ok($execute_impl::<_, _, true, false>),
            (false, true) => Ok($execute_impl::<_, _, false, true>),
            (false, false) => Ok($execute_impl::<_, _, false, false>),
        }
    };
}

impl<F> InterpreterExecutor<F> for LoadSignExtendExecutor<{ RV32_REGISTER_NUM_LIMBS }>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        size_of::<LoadSignExtendPreCompute>()
    }

    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut LoadSignExtendPreCompute = data.borrow_mut();
        let (is_loadb, enabled) = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, is_loadb, enabled)
    }
}

impl<F> InterpreterMeteredExecutor<F> for LoadSignExtendExecutor<{ RV32_REGISTER_NUM_LIMBS }>
where
    F: PrimeField32,
{
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<LoadSignExtendPreCompute>>()
    }

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
        let pre_compute: &mut E2PreCompute<LoadSignExtendPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;
        let (is_loadb, enabled) = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        dispatch!(execute_e2_handler, is_loadb, enabled)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_LOADB: bool,
    const ENABLED: bool,
>(
    pre_compute: &LoadSignExtendPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pc = exec_state.pc();
    let fp = exec_state.memory.fp::<F>();
    let rs1_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
        exec_state.vm_read(RV32_REGISTER_AS, fp + pre_compute.b);
    let rs1_val = u32::from_le_bytes(rs1_bytes);
    let ptr_val = rs1_val.wrapping_add(pre_compute.imm_extended);
    // sign_extend([r32{c,g}(b):2]_e)`
    debug_assert!(ptr_val < (1 << POINTER_MAX_BITS));
    let shift_amount = ptr_val % 4;
    let ptr_val = ptr_val - shift_amount; // aligned ptr

    let read_data: [u8; RV32_REGISTER_NUM_LIMBS] = exec_state.vm_read(pre_compute.e, ptr_val);

    let write_data = if IS_LOADB {
        let byte = read_data[shift_amount as usize];
        let sign_extended = (byte as i8) as i32;
        sign_extended.to_le_bytes()
    } else {
        if shift_amount != 0 && shift_amount != 2 {
            let err = ExecutionError::Fail {
                pc,
                msg: "LoadSignExtend invalid shift amount",
            };
            return Err(err);
        }
        let half: [u8; 2] = array::from_fn(|i| read_data[shift_amount as usize + i]);
        (i16::from_le_bytes(half) as i32).to_le_bytes()
    };

    if ENABLED {
        exec_state.vm_write(RV32_REGISTER_AS, fp + pre_compute.a, &write_data);
    }

    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const IS_LOADB: bool,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    unsafe {
        let pre_compute: &LoadSignExtendPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<LoadSignExtendPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, IS_LOADB, ENABLED>(pre_compute, exec_state)
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const IS_LOADB: bool,
    const ENABLED: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    unsafe {
        let pre_compute: &E2PreCompute<LoadSignExtendPreCompute> = std::slice::from_raw_parts(
            pre_compute,
            size_of::<E2PreCompute<LoadSignExtendPreCompute>>(),
        )
        .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, IS_LOADB, ENABLED>(&pre_compute.data, exec_state)
    }
}
