use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::*,
    system::memory::{
        MemoryAuxColsFactory,
        online::{GuestMemory, TracingMemory},
    },
};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_crush_transpiler::JumpOpcode;
use openvm_instructions::{
    LocalOpcode, instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;
use strum::EnumCount;

use super::core::{JumpCoreFiller, JumpCoreRecord};
use crate::adapters::{JumpAdapterExecutor, JumpAdapterFiller, RV32_REGISTER_NUM_LIMBS};
use openvm_circuit::arch::AdapterTraceFiller;

/// Executor for the JUMP chip (preflight).
#[derive(Clone)]
pub struct JumpExecutor {
    pub adapter: JumpAdapterExecutor,
    pub offset: usize,
}

impl JumpExecutor {
    pub fn new(adapter: JumpAdapterExecutor, offset: usize) -> Self {
        Self { adapter, offset }
    }
}

impl<F, RA> PreflightExecutor<F, RA> for JumpExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<
            'buf,
            EmptyAdapterCoreLayout<F, JumpAdapterExecutor>,
            (
                <JumpAdapterExecutor as AdapterTraceExecutor<F>>::RecordMut<'buf>,
                &'buf mut JumpCoreRecord,
            ),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", JumpOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { opcode, a: imm, .. } = instruction;
        let local_opcode = JumpOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        <JumpAdapterExecutor as AdapterTraceExecutor<F>>::start(
            *state.pc,
            *state.fp,
            state.memory,
            &mut adapter_record,
        );

        let [rs_val] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);

        core_record.rs_val = rs_val;
        core_record.imm = imm.as_canonical_u32();
        core_record.local_opcode = local_opcode as u8;

        // Compute the register value (composed from limbs).
        let reg_value = u32::from_le_bytes(rs_val);

        match local_opcode {
            JumpOpcode::JUMP => {
                *state.pc = imm.as_canonical_u32();
            }
            JumpOpcode::SKIP => {
                // PC += (reg_value + 1) * DEFAULT_PC_STEP
                // The +1 accounts for the natural PC increment that crush's interpreter
                // applies after JumpOffset. Without it, offset=0 would loop forever.
                *state.pc = state
                    .pc
                    .wrapping_add(reg_value.wrapping_add(1).wrapping_mul(DEFAULT_PC_STEP));
            }
            JumpOpcode::JUMP_IF => {
                if reg_value != 0 {
                    *state.pc = imm.as_canonical_u32();
                } else {
                    *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
                }
            }
            JumpOpcode::JUMP_IF_ZERO => {
                if reg_value == 0 {
                    *state.pc = imm.as_canonical_u32();
                } else {
                    *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
                }
            }
        }

        Ok(())
    }
}

/// Filler for the JUMP chip (trace generation).
#[derive(derive_new::new)]
pub struct JumpFiller {
    pub adapter: JumpAdapterFiller,
    pub core: JumpCoreFiller,
}

impl<F: PrimeField32> TraceFiller<F> for JumpFiller {
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, core_row) = unsafe {
            row_slice.split_at_mut_unchecked(<JumpAdapterFiller as AdapterTraceFiller<F>>::WIDTH)
        };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        self.core.fill_trace_row(core_row);
    }
}

// ==================== InterpreterExecutor ====================

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct JumpPreCompute {
    /// Immediate value (to_pc for JUMP/JUMP_IF/JUMP_IF_ZERO, unused for SKIP).
    imm: u32,
    /// Register pointer for condition/offset register.
    rs_ptr: u32,
}

impl JumpExecutor {
    /// Fills the precomputed data and returns the local opcode.
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut JumpPreCompute,
    ) -> Result<JumpOpcode, StaticProgramError> {
        let Instruction { opcode, a, b, .. } = inst;

        let local_opcode_idx = opcode.local_opcode_idx(self.offset);
        if local_opcode_idx >= JumpOpcode::COUNT {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        *data = JumpPreCompute {
            imm: a.as_canonical_u32(),
            rs_ptr: b.as_canonical_u32(),
        };
        Ok(JumpOpcode::from_usize(local_opcode_idx))
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $local_opcode:expr) => {
        match $local_opcode {
            JumpOpcode::JUMP => Ok($execute_impl::<_, _, { JumpOpcode::JUMP as u8 }>),
            JumpOpcode::SKIP => Ok($execute_impl::<_, _, { JumpOpcode::SKIP as u8 }>),
            JumpOpcode::JUMP_IF => Ok($execute_impl::<_, _, { JumpOpcode::JUMP_IF as u8 }>),
            JumpOpcode::JUMP_IF_ZERO => {
                Ok($execute_impl::<_, _, { JumpOpcode::JUMP_IF_ZERO as u8 }>)
            }
        }
    };
}

impl<F: PrimeField32> InterpreterExecutor<F> for JumpExecutor {
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<JumpPreCompute>()
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
        let data: &mut JumpPreCompute = data.borrow_mut();
        let local_opcode = self.pre_compute_impl(pc, inst, data)?;
        dispatch!(execute_e1_handler, local_opcode)
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for JumpExecutor {
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<JumpPreCompute>>()
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
        let data: &mut E2PreCompute<JumpPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        let local_opcode = self.pre_compute_impl(pc, inst, &mut data.data)?;
        dispatch!(execute_e2_handler, local_opcode)
    }
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const OPCODE: u8>(
    pre_compute: &JumpPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let fp = exec_state.fp();
    let pc = exec_state.pc();

    // Always read the condition/offset register relative to FP.
    // For JUMP (b=0), this reads reg[fp+0]; the core chip ignores the value.
    let rs: [u8; RV32_REGISTER_NUM_LIMBS] =
        exec_state.vm_read(RV32_REGISTER_AS, fp + pre_compute.rs_ptr);
    let reg_value = u32::from_le_bytes(rs);

    let new_pc = match OPCODE {
        x if x == JumpOpcode::JUMP as u8 => pre_compute.imm,
        x if x == JumpOpcode::SKIP as u8 => {
            pc.wrapping_add(reg_value.wrapping_add(1).wrapping_mul(DEFAULT_PC_STEP))
        }
        x if x == JumpOpcode::JUMP_IF as u8 => {
            if reg_value != 0 {
                pre_compute.imm
            } else {
                pc.wrapping_add(DEFAULT_PC_STEP)
            }
        }
        x if x == JumpOpcode::JUMP_IF_ZERO as u8 => {
            if reg_value == 0 {
                pre_compute.imm
            } else {
                pc.wrapping_add(DEFAULT_PC_STEP)
            }
        }
        _ => unreachable!(),
    };

    exec_state.set_pc(new_pc);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const OPCODE: u8>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &JumpPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<JumpPreCompute>()).borrow();
        execute_e12_impl::<F, CTX, OPCODE>(pre_compute, exec_state);
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait, const OPCODE: u8>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<JumpPreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<JumpPreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, OPCODE>(&pre_compute.data, exec_state);
    }
}
