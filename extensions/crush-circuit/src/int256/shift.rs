use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_crush_transpiler::Shift256Opcode;
use openvm_instructions::{
    LocalOpcode,
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
};
use openvm_rv32im_circuit::ShiftExecutor;
use openvm_rv32im_transpiler::ShiftOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::memory_config::FpMemory;

use crate::int256::{
    AluAdapterExecutor, INT256_NUM_LIMBS, Rv32Shift256Executor,
    common::{bytes_to_u64_array, read_int256, u64_array_to_bytes, write_int256},
};

impl Rv32Shift256Executor {
    pub fn new(adapter: AluAdapterExecutor, offset: usize) -> Self {
        Self(ShiftExecutor::new(adapter, offset))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct ShiftPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:ident) => {
        Ok(match $local_opcode {
            ShiftOpcode::SLL => $execute_impl::<_, _, SllOp>,
            ShiftOpcode::SRA => $execute_impl::<_, _, SraOp>,
            ShiftOpcode::SRL => $execute_impl::<_, _, SrlOp>,
        })
    };
}

impl<F: PrimeField32> InterpreterExecutor<F> for Rv32Shift256Executor {
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
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut ShiftPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotExecutor<F> for Rv32Shift256Executor {}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for Rv32Shift256Executor {
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
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<ShiftPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for Rv32Shift256Executor {}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: ShiftOp>(
    pre_compute: &ShiftPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    // Register addresses are frame-pointer relative.
    let fp = exec_state.memory.fp::<F>();
    let rs1_ptr = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, fp + pre_compute.b as u32);
    let rs2_ptr = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, fp + pre_compute.c as u32);
    let rd_ptr = exec_state.vm_read::<u8, 4>(RV32_REGISTER_AS, fp + pre_compute.a as u32);
    let rs1 = read_int256(exec_state, RV32_MEMORY_AS, u32::from_le_bytes(rs1_ptr));
    let rs2 = read_int256(exec_state, RV32_MEMORY_AS, u32::from_le_bytes(rs2_ptr));
    let rd = OP::compute(rs1, rs2);
    write_int256(exec_state, RV32_MEMORY_AS, u32::from_le_bytes(rd_ptr), &rd);
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, OP: ShiftOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &ShiftPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<ShiftPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, OP>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, OP: ShiftOp>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<ShiftPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<ShiftPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<F, CTX, OP>(&pre_compute.data, exec_state);
}

impl Rv32Shift256Executor {
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut ShiftPreCompute,
    ) -> Result<ShiftOpcode, StaticProgramError> {
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
        if d.as_canonical_u32() != RV32_REGISTER_AS || e_u32 != RV32_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        *data = ShiftPreCompute {
            a: a.as_canonical_u32() as u8,
            b: b.as_canonical_u32() as u8,
            c: c.as_canonical_u32() as u8,
        };
        let local_opcode =
            ShiftOpcode::from_usize(opcode.local_opcode_idx(Shift256Opcode::CLASS_OFFSET));
        Ok(local_opcode)
    }
}

trait ShiftOp {
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS];
}
struct SllOp;
struct SrlOp;
struct SraOp;
impl ShiftOp for SllOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        let rs1_u64: [u64; 4] = bytes_to_u64_array(rs1);
        let rs2_u64: [u64; 4] = bytes_to_u64_array(rs2);
        let mut rd = [0u64; 4];
        // Only use the first 8 bits.
        let shift = (rs2_u64[0] & 0xff) as u32;
        let index_offset = (shift / u64::BITS) as usize;
        let bit_offset = shift % u64::BITS;
        let mut carry = 0u64;
        for i in index_offset..4 {
            let curr = rs1_u64[i - index_offset];
            rd[i] = (curr << bit_offset) + carry;
            if bit_offset > 0 {
                carry = curr >> (u64::BITS - bit_offset);
            }
        }
        u64_array_to_bytes(rd)
    }
}
impl ShiftOp for SrlOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        // Logical right shift - fill with 0
        shift_right(rs1, rs2, 0)
    }
}
impl ShiftOp for SraOp {
    #[inline(always)]
    fn compute(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> [u8; INT256_NUM_LIMBS] {
        // Arithmetic right shift - fill with sign bit
        if rs1[INT256_NUM_LIMBS - 1] & 0x80 > 0 {
            shift_right(rs1, rs2, u64::MAX)
        } else {
            shift_right(rs1, rs2, 0)
        }
    }
}

#[inline(always)]
fn shift_right(
    rs1: [u8; INT256_NUM_LIMBS],
    rs2: [u8; INT256_NUM_LIMBS],
    init_value: u64,
) -> [u8; INT256_NUM_LIMBS] {
    let rs1_u64: [u64; 4] = bytes_to_u64_array(rs1);
    let rs2_u64: [u64; 4] = bytes_to_u64_array(rs2);
    let mut rd = [init_value; 4];
    let shift = (rs2_u64[0] & 0xff) as u32;
    let index_offset = (shift / u64::BITS) as usize;
    let bit_offset = shift % u64::BITS;
    let mut carry = if bit_offset > 0 {
        init_value << (u64::BITS - bit_offset)
    } else {
        0
    };
    for i in (index_offset..4).rev() {
        let curr = rs1_u64[i];
        rd[i - index_offset] = (curr >> bit_offset) + carry;
        if bit_offset > 0 {
            carry = curr << (u64::BITS - bit_offset);
        }
    }
    u64_array_to_bytes(rd)
}
