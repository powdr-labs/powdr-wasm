use crate::adapters::tracing_read_fp;
use crate::adapters::{decompose, tracing_write};
use crate::air::Const32AdapterAirCol;
use crate::memory_config::FpMemory;
use itertools::Itertools;
use openvm_circuit::arch::*;
use openvm_circuit::system::memory::offline_checker::MemoryReadAuxRecord;
use openvm_circuit::system::memory::offline_checker::MemoryWriteBytesAuxRecord;
use openvm_circuit::system::memory::online::TracingMemory;
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_stark_backend::p3_field::PrimeField32;
use std::borrow::{Borrow, BorrowMut};
// Minimal executor for CONST32 - no computation needed, just write immediate to register
#[derive(Clone, derive_new::new)]
pub struct Const32Executor<const NUM_LIMBS: usize> {
    pub offset: usize,
}

// PreCompute struct for CONST32
#[repr(C)]
#[derive(AlignedBytesBorrow, Clone, Copy)]
struct Const32PreCompute {
    target_reg: u32,
    imm: u32,
}

impl<const NUM_LIMBS: usize> Const32Executor<NUM_LIMBS> {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        inst: &Instruction<F>,
        data: &mut Const32PreCompute,
    ) {
        let Instruction { a, b, c, .. } = *inst;
        let imm = (b.as_canonical_u32() & 0xFFFF) | ((c.as_canonical_u32() & 0xFFFF) << 16);
        *data = Const32PreCompute {
            target_reg: a.as_canonical_u32(),
            imm,
        };
    }
}

impl<F, RA> PreflightExecutor<F, RA> for Const32Executor<RV32_REGISTER_NUM_LIMBS>
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, EmptyMultiRowLayout, &'buf mut Const32Record>,
{
    fn get_opcode_name(&self, _opcode: usize) -> String {
        "CONST32".to_string()
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { a, b, c, .. } = instruction;
        let record = state
            .ctx
            .alloc(EmptyMultiRowLayout::new(EmptyMultiRowMetadata::new()));

        // Extract immediates (16-bit values in b and c)
        let imm_lo = b.as_canonical_u32() & 0xFFFF;
        let imm_hi = c.as_canonical_u32() & 0xFFFF;

        // Combine to form 32-bit immediate
        let imm = (imm_hi << 16) | imm_lo;

        // Decompose into limbs
        let value: [F; RV32_REGISTER_NUM_LIMBS] = decompose(imm);
        let value_bytes = value.map(|x| x.as_canonical_u32() as u8);
        record.from_pc = *state.pc;
        record.from_timestamp = state.memory.timestamp;
        record.rd_ptr = a.as_canonical_u32();
        record.imm = imm;

        record.fp = tracing_read_fp::<F>(state.memory, &mut record.fp_read_aux.prev_timestamp);
        tracing_write(
            state.memory,
            RV32_REGISTER_AS,
            record.rd_ptr + record.fp,
            value_bytes,
            &mut record.writes_aux.prev_timestamp,
            &mut record.writes_aux.prev_data,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

// InterpreterExecutor implementation
impl<F> InterpreterExecutor<F> for Const32Executor<RV32_REGISTER_NUM_LIMBS>
where
    F: PrimeField32,
{
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<Const32PreCompute>()
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
        let data: &mut Const32PreCompute = data.borrow_mut();
        self.pre_compute_impl(inst, data);

        Ok(execute_e1_handler::<F, Ctx, RV32_REGISTER_NUM_LIMBS>)
    }
}

impl<F> InterpreterMeteredExecutor<F> for Const32Executor<RV32_REGISTER_NUM_LIMBS>
where
    F: PrimeField32,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<Const32PreCompute>>()
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
        let data: &mut E2PreCompute<Const32PreCompute> = data.borrow_mut();
        data.chip_idx = chip_idx as u32;
        self.pre_compute_impl(inst, &mut data.data);

        Ok(execute_e2_handler::<F, Ctx, RV32_REGISTER_NUM_LIMBS>)
    }
}

// Execute function for CONST32
unsafe fn execute_e12_impl<F: PrimeField32, Ctx: ExecutionCtxTrait, const NUM_LIMBS: usize>(
    pre_compute: &Const32PreCompute,
    exec_state: &mut VmExecState<F, openvm_circuit::system::memory::online::GuestMemory, Ctx>,
) {
    let fp = exec_state.memory.fp::<F>();

    let imm_bytes: [u8; NUM_LIMBS] = std::array::from_fn(|i| (pre_compute.imm >> (8 * i)) as u8);
    exec_state.vm_write::<u8, NUM_LIMBS>(RV32_REGISTER_AS, fp + pre_compute.target_reg, &imm_bytes);

    // Increment PC
    let next_pc = exec_state.pc().wrapping_add(DEFAULT_PC_STEP);
    exec_state.set_pc(next_pc);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const NUM_LIMBS: usize>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, openvm_circuit::system::memory::online::GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &Const32PreCompute =
            std::slice::from_raw_parts(pre_compute, size_of::<Const32PreCompute>()).borrow();
        execute_e12_impl::<F, CTX, NUM_LIMBS>(pre_compute, exec_state);
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const NUM_LIMBS: usize,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, openvm_circuit::system::memory::online::GuestMemory, CTX>,
) {
    unsafe {
        let pre_compute: &E2PreCompute<Const32PreCompute> =
            std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<Const32PreCompute>>())
                .borrow();
        exec_state
            .ctx
            .on_height_change(pre_compute.chip_idx as usize, 1);
        execute_e12_impl::<F, CTX, NUM_LIMBS>(&pre_compute.data, exec_state);
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow)]
pub struct Const32Record {
    pub from_pc: u32,
    pub fp: u32,
    pub from_timestamp: u32,

    pub rd_ptr: u32,
    pub imm: u32,
    pub fp_read_aux: MemoryReadAuxRecord,
    pub writes_aux: MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS>,
}

#[derive(derive_new::new)]
pub struct Const32Filler<const NUM_LIMBS: usize> {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
}

impl<F: PrimeField32, const NUM_LIMBS: usize> TraceFiller<F> for Const32Filler<NUM_LIMBS> {
    fn fill_trace_row(
        &self,
        mem_helper: &openvm_circuit::system::memory::MemoryAuxColsFactory<F>,
        mut row_slice: &mut [F],
    ) {
        let record: &Const32Record = unsafe { get_record_from_slice(&mut row_slice, ()) };
        let cols: &mut Const32AdapterAirCol<F, NUM_LIMBS> = row_slice.borrow_mut();

        // fp_read_aux: fill timestamp proof for FP read at from_timestamp + 0
        mem_helper.fill(
            record.fp_read_aux.prev_timestamp,
            record.from_timestamp,
            cols.fp_read_aux.as_mut(),
        );

        // write_aux: set prev_data and fill timestamp proof
        // Write happens at from_timestamp + 1 (after FP read at from_timestamp + 0)
        cols.write_aux.set_prev_data(std::array::from_fn(|i| {
            F::from_u8(record.writes_aux.prev_data[i])
        }));
        mem_helper.fill(
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 1,
            cols.write_aux.as_mut(),
        );

        // imm_limbs: decompose the immediate into limbs and range-check
        let imm = record.imm;
        let mask = (1u32 << RV32_CELL_BITS) - 1;
        let imm_limbs_u32 = std::array::from_fn(|i| (imm >> (RV32_CELL_BITS * i)) & mask);
        cols.imm_limbs = imm_limbs_u32.map(F::from_u32);
        for (lo, hi) in imm_limbs_u32.iter().copied().tuples() {
            self.bitwise_lookup_chip.request_range(lo, hi);
        }

        // rd_ptr
        cols.rd_ptr = F::from_u32(record.rd_ptr);

        // from_state
        cols.from_state.timestamp = F::from_u32(record.from_timestamp);
        cols.from_state.fp = F::from_u32(record.fp);
        cols.from_state.pc = F::from_u32(record.from_pc);

        // is_valid
        cols.is_valid = F::ONE;
    }
}
