use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller, BasicAdapterInterface,
        ImmInstruction, VmAdapterAir, get_record_from_slice,
    },
    system::memory::{
        MemoryAuxColsFactory,
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord},
        online::TracingMemory,
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS,
};
use openvm_stark_backend::{
    ColumnsAir,
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};
use struct_reflection::{StructReflection, StructReflectionHelper};

use openvm_circuit::arch::ExecutionBridge;

use crate::execution::ExecutionState;

use super::{RV32_REGISTER_NUM_LIMBS, fp_addr, fp_block, reg_addr, tracing_read, tracing_read_fp};

/// Trace columns for the JUMP adapter.
///
/// Always reads one register (the condition/offset register, field b) relative to FP.
/// For JUMP (unconditional), b=0, so we read reg[fp+0] and the core chip ignores the value.
/// No writes.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct JumpAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    /// The condition/offset register pointer (field b of instruction).
    pub rs_ptr: T,
    pub fp_read_aux: MemoryReadAuxCols<T>,
    pub rs_read_aux: MemoryReadAuxCols<T>,
}

/// AIR for the JUMP adapter.
///
/// Handles FP-relative register reads for JUMP instructions.
/// Always reads 1 register, writes nothing.
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct JumpAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for JumpAdapterAir {
    fn width(&self) -> usize {
        JumpAdapterCols::<F>::width()
    }
}

impl<F: Field> ColumnsAir<F> for JumpAdapterAir {
    fn columns(&self) -> Option<Vec<String>> {
        JumpAdapterCols::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for JumpAdapterAir {
    // 1 read of RV32_REGISTER_NUM_LIMBS limbs, 0 writes
    type Interface =
        BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, 1, 0, RV32_REGISTER_NUM_LIMBS, 0>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &JumpAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Read FP
        self.memory_bridge
            .read(
                fp_addr::<AB::F>(),
                fp_block::<AB::Expr>(local.from_state.fp.into()),
                timestamp_pp(),
                &local.fp_read_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // Always read the condition/offset register relative to FP.
        // For JUMP (b=0), this reads reg[fp+0]; the core chip ignores the value.
        self.memory_bridge
            .read(
                reg_addr(local.rs_ptr + local.from_state.fp),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.rs_read_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    ctx.instruction.immediate.clone(),
                    local.rs_ptr.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ONE,  // f
                    AB::Expr::ZERO, // g: imm sign
                ],
                local.from_state.into(),
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &JumpAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

/// Record for the JUMP adapter (written during preflight, read during trace fill).
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct JumpAdapterRecord {
    pub from_pc: u32,
    pub fp: u32,
    pub from_timestamp: u32,
    pub rs_ptr: u32,
    pub fp_read_aux: MemoryReadAuxRecord,
    pub rs_read_aux: MemoryReadAuxRecord,
}

/// Executor for the JUMP adapter (preflight).
#[derive(Clone, Default)]
pub struct JumpAdapterExecutor {}

impl<F: PrimeField32> AdapterTraceExecutor<F> for JumpAdapterExecutor {
    const WIDTH: usize = size_of::<JumpAdapterCols<u8>>();
    type ReadData = [[u8; RV32_REGISTER_NUM_LIMBS]; 1];
    type WriteData = ();
    type RecordMut<'a> = &'a mut JumpAdapterRecord;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut &mut JumpAdapterRecord) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut &mut JumpAdapterRecord,
    ) -> Self::ReadData {
        let &Instruction { b, .. } = instruction;

        // HACK: The frame pointer fetch must happen exactly once before the first register access.
        // We can do it here unconditionally because Self::ReadData has length 1.
        record.fp = tracing_read_fp::<F>(memory, &mut record.fp_read_aux.prev_timestamp);

        let b_val = b.as_canonical_u32();
        record.rs_ptr = b_val;

        // Always read the register (even for JUMP where b=0).
        let rs = tracing_read(
            memory,
            RV32_REGISTER_AS,
            b_val + record.fp,
            &mut record.rs_read_aux.prev_timestamp,
        );

        [rs]
    }

    #[inline(always)]
    fn write(
        &self,
        _memory: &mut TracingMemory,
        _instruction: &Instruction<F>,
        _data: Self::WriteData,
        _record: &mut Self::RecordMut<'_>,
    ) {
        // JUMP instructions don't write any registers.
    }
}

/// Trace filler for the JUMP adapter.
#[derive(derive_new::new)]
pub struct JumpAdapterFiller;

impl<F: PrimeField32> AdapterTraceFiller<F> for JumpAdapterFiller {
    const WIDTH: usize = size_of::<JumpAdapterCols<u8>>();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        let record: &JumpAdapterRecord = unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut JumpAdapterCols<F> = adapter_row.borrow_mut();

        // We must assign in reverse order.

        // rs read (timestamp = from_timestamp + 1, after fp read)
        let timestamp = record.from_timestamp + 1;
        mem_helper.fill(
            record.rs_read_aux.prev_timestamp,
            timestamp,
            adapter_row.rs_read_aux.as_mut(),
        );

        // fp read (timestamp = from_timestamp)
        mem_helper.fill(
            record.fp_read_aux.prev_timestamp,
            record.from_timestamp,
            adapter_row.fp_read_aux.as_mut(),
        );

        adapter_row.rs_ptr = F::from_u32(record.rs_ptr);
        adapter_row.from_state.timestamp = F::from_u32(record.from_timestamp);
        adapter_row.from_state.fp = F::from_u32(record.fp);
        adapter_row.from_state.pc = F::from_u32(record.from_pc);
    }
}
