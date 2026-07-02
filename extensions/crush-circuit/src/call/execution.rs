use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{
    arch::*,
    system::memory::online::{GuestMemory, TracingMemory},
};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_crush_transpiler::CallOpcode;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS, LocalOpcode,
    VmOpcode,
};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::call::{CallAdapterWrite, CallData};
use crate::adapters::RV32_REGISTER_NUM_LIMBS;

use super::core::CallCoreRecord;

/// Call executor. Wraps the adapter executor.
#[derive(Clone, Copy)]
pub struct CallExecutor<A> {
    pub adapter: A,
    pub offset: usize,
}

impl<A> CallExecutor<A> {
    pub fn new(adapter: A, offset: usize) -> Self {
        Self { adapter, offset }
    }
}

impl<A> std::ops::Deref for CallExecutor<A> {
    type Target = A;
    fn deref(&self) -> &Self::Target {
        &self.adapter
    }
}

/// Pre-computed data for Call instruction execution (E1/E2 stages).
#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
pub(super) struct CallPreCompute {
    pub to_fp_operand: u32,
    pub save_fp_ptr: u32,
    pub save_pc_ptr: u32,
    pub to_pc_operand: u32,
    pub opcode: u8,
}

impl<F, A, RA> PreflightExecutor<F, RA> for CallExecutor<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceExecutor<F, ReadData = CallData<u8>, WriteData = CallAdapterWrite<u8>>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut CallCoreRecord),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        let local_idx = VmOpcode::from_usize(opcode).local_opcode_idx(CallOpcode::CLASS_OFFSET);
        format!("{:?}", CallOpcode::from_usize(local_idx))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let local_idx = instruction.opcode.local_opcode_idx(self.offset);
        let opcode = CallOpcode::from_usize(local_idx);

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(
            *state.pc,
            *state.extra_regs,
            state.memory,
            &mut adapter_record,
        );

        // Read through the adapter: [new_fp_bytes, to_pc_bytes]
        // new_fp_bytes is only valid for RET (register read); zeros for CALL/CALL_INDIRECT
        let CallData {
            fp_data: new_fp_bytes,
            pc_data: to_pc_bytes,
        } = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);

        // Old FP comes from the VM execution state (no longer stored in memory).
        let old_fp_val = state.extra_regs[0];

        // Compute actual new FP:
        // CALL/CALL_INDIRECT: new_fp = old_fp + immediate offset (d operand)
        // RET: new_fp = absolute FP from register
        let new_fp = match opcode {
            CallOpcode::CALL | CallOpcode::CALL_INDIRECT => {
                let nfp = old_fp_val + instruction.d.as_canonical_u32();
                debug_assert!(nfp >= old_fp_val, "fp + offset overflows u32");
                nfp
            }
            CallOpcode::RET => u32::from_le_bytes(new_fp_bytes),
        };

        // Fill core record
        core_record.new_fp_data = new_fp_bytes;
        core_record.to_pc_data = to_pc_bytes;
        core_record.old_fp_data = old_fp_val.to_le_bytes();
        core_record.return_pc_data = (*state.pc + DEFAULT_PC_STEP).to_le_bytes();
        core_record.local_opcode = opcode as u8;

        // Compute write data
        let save_fp_bytes = old_fp_val.to_le_bytes();
        let save_pc_bytes = (*state.pc + DEFAULT_PC_STEP).to_le_bytes();

        // Write through the adapter
        self.adapter.write(
            state.memory,
            instruction,
            CallAdapterWrite {
                save_fp: save_fp_bytes,
                save_pc: save_pc_bytes,
                new_fp,
            },
            &mut adapter_record,
        );

        // Determine the target PC
        let to_pc = match opcode {
            CallOpcode::RET | CallOpcode::CALL_INDIRECT => u32::from_le_bytes(to_pc_bytes),
            CallOpcode::CALL => instruction.c.as_canonical_u32(),
        };

        *state.pc = to_pc;
        // FP is carried in the VM state; update it for the next instruction.
        state.extra_regs[0] = new_fp;

        Ok(())
    }
}

impl<F: PrimeField32, A> InterpreterExecutor<F> for CallExecutor<A> {
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<CallPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let data: &mut CallPreCompute = data.borrow_mut();
        let local_idx = inst.opcode.local_opcode_idx(CallOpcode::CLASS_OFFSET);
        let opcode = CallOpcode::from_usize(local_idx);

        data.to_fp_operand = inst.d.as_canonical_u32();
        data.save_fp_ptr = inst.b.as_canonical_u32();
        data.save_pc_ptr = inst.a.as_canonical_u32();
        data.to_pc_operand = inst.c.as_canonical_u32();
        data.opcode = opcode as u8;

        Ok(execute_e1_handler::<F, Ctx>)
    }
}

impl<F: PrimeField32, A> InterpreterMeteredExecutor<F> for CallExecutor<A> {
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<CallPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let data: &mut E2PreCompute<CallPreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;

        let local_idx = inst.opcode.local_opcode_idx(CallOpcode::CLASS_OFFSET);
        let opcode = CallOpcode::from_usize(local_idx);

        data.data.to_fp_operand = inst.d.as_canonical_u32();
        data.data.save_fp_ptr = inst.b.as_canonical_u32();
        data.data.save_pc_ptr = inst.a.as_canonical_u32();
        data.data.to_pc_operand = inst.c.as_canonical_u32();
        data.data.opcode = opcode as u8;

        Ok(execute_e2_handler::<F, Ctx>)
    }
}

#[inline(always)]
unsafe fn execute_call_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre: &CallPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    // TODO: Instead of dispatching at runtime, we should do it at compile-time!
    let opcode = CallOpcode::from_repr(pre.opcode as usize).unwrap();
    let fp = exec_state.extra_regs()[0];

    // Compute new FP:
    // CALL/CALL_INDIRECT: new_fp = old_fp + immediate offset (to_fp_operand)
    // RET: new_fp = absolute FP from register (to_fp_operand is register pointer)
    let new_fp = match opcode {
        CallOpcode::CALL | CallOpcode::CALL_INDIRECT => {
            let nfp = fp + pre.to_fp_operand;
            debug_assert!(nfp >= fp, "fp + offset overflows u32");
            nfp
        }
        CallOpcode::RET => {
            let fp_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
                exec_state.vm_read(RV32_REGISTER_AS, fp + pre.to_fp_operand);
            u32::from_le_bytes(fp_bytes)
        }
    };

    // Read target PC from register (for RET, CALL_INDIRECT) or use immediate (for CALL)
    let to_pc = match opcode {
        CallOpcode::RET | CallOpcode::CALL_INDIRECT => {
            let pc_bytes: [u8; RV32_REGISTER_NUM_LIMBS] =
                exec_state.vm_read(RV32_REGISTER_AS, fp + pre.to_pc_operand);
            u32::from_le_bytes(pc_bytes)
        }
        CallOpcode::CALL => pre.to_pc_operand,
    };

    let old_pc = exec_state.pc();
    let old_fp = fp;

    // Save old FP and return PC to registers in new frame (for CALL, CALL_INDIRECT)
    match opcode {
        CallOpcode::CALL | CallOpcode::CALL_INDIRECT => {
            exec_state.vm_write::<u8, RV32_REGISTER_NUM_LIMBS>(
                RV32_REGISTER_AS,
                new_fp + pre.save_fp_ptr,
                &old_fp.to_le_bytes(),
            );
            let return_pc = old_pc + DEFAULT_PC_STEP;
            exec_state.vm_write::<u8, RV32_REGISTER_NUM_LIMBS>(
                RV32_REGISTER_AS,
                new_fp + pre.save_pc_ptr,
                &return_pc.to_le_bytes(),
            );
        }
        _ => {}
    }

    // Set new FP
    exec_state.set_extra_regs([new_fp]);

    // Set new PC
    exec_state.set_pc(to_pc);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre: &CallPreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<CallPreCompute>()).borrow();
        execute_call_impl::<F, CTX>(pre, exec_state);
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    unsafe {
        let pre: &E2PreCompute<CallPreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<CallPreCompute>>())
                .borrow();
        exec_state.ctx.on_height_change(pre.chip_idx as usize, 1);
        execute_call_impl::<F, CTX>(&pre.data, exec_state);
    }
}
