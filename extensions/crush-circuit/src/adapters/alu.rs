use std::{
    borrow::{Borrow, BorrowMut},
    sync::{Arc, Mutex},
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller, BasicAdapterInterface,
        MinimalInstruction, VmAdapterAir, get_record_from_slice,
    },
    system::memory::{
        MemoryAddress, MemoryAuxColsFactory,
        offline_checker::{
            MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
            MemoryWriteBytesAuxRecord,
        },
        online::TracingMemory,
    },
};
use openvm_circuit_primitives::{
    AlignedBytesBorrow,
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    utils::not,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
};
use openvm_stark_backend::{
    ColumnsAir,
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};
use struct_reflection::{StructReflection, StructReflectionHelper};

use openvm_circuit::arch::ExecutionBridge;

use crate::execution::ExecutionState;

use super::{
    RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, W32_REG_OPS, fp_addr, reg_addr, tracing_read,
    tracing_read_fp, tracing_read_imm, tracing_write,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct BaseAluAdapterColsDifferentInputsOutputs<
    T,
    const NUM_READ_OPS: usize,
    const NUM_WRITE_OPS: usize,
> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    /// Pointer if rs2 was a read, immediate value otherwise
    pub rs2: T,
    /// 1 if rs2 was a read, 0 if an immediate
    pub rs2_as: T,
    pub fp_read_aux: MemoryReadAuxCols<T>,
    pub rs1_reads_aux: [MemoryReadAuxCols<T>; NUM_READ_OPS],
    pub rs2_reads_aux: [MemoryReadAuxCols<T>; NUM_READ_OPS],
    pub writes_aux: [MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>; NUM_WRITE_OPS],
}

/// Reads instructions of the form OP a, b, c, d, e where \[a:N\]_d = \[b:N\]_d op \[c:N\]_e.
/// Operand d can only be 1, and e can be either 1 (for register reads) or 0 (when c
/// is an immediate).
///
/// `NUM_READ_OPS` controls how many 4-byte chunks are read per operand.
/// `NUM_WRITE_OPS` controls how many 4-byte chunks are written for the result.
/// For most 64-bit ops both are 2, but for comparisons (LessThan64) the result is
/// only 32 bits so `NUM_WRITE_OPS=1` while `NUM_READ_OPS=2`.
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct BaseAluAdapterAirDifferentInputsOutputs<
    const NUM_LIMBS: usize,
    const NUM_READ_OPS: usize,
    const NUM_WRITE_OPS: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl<F: Field, const NUM_LIMBS: usize, const NUM_READ_OPS: usize, const NUM_WRITE_OPS: usize>
    BaseAir<F> for BaseAluAdapterAirDifferentInputsOutputs<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS>
{
    fn width(&self) -> usize {
        BaseAluAdapterColsDifferentInputsOutputs::<F, NUM_READ_OPS, NUM_WRITE_OPS>::width()
    }
}

impl<F: Field, const NUM_LIMBS: usize, const NUM_READ_OPS: usize, const NUM_WRITE_OPS: usize>
    ColumnsAir<F>
    for BaseAluAdapterAirDifferentInputsOutputs<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS>
{
    fn columns(&self) -> Option<Vec<String>> {
        BaseAluAdapterColsDifferentInputsOutputs::<F, NUM_READ_OPS, NUM_WRITE_OPS>::struct_reflection()
    }
}

impl<
    AB: InteractionBuilder,
    const NUM_LIMBS: usize,
    const NUM_READ_OPS: usize,
    const NUM_WRITE_OPS: usize,
> VmAdapterAir<AB>
    for BaseAluAdapterAirDifferentInputsOutputs<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS>
{
    type Interface =
        BasicAdapterInterface<AB::Expr, MinimalInstruction<AB::Expr>, 2, 1, NUM_LIMBS, NUM_LIMBS>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &BaseAluAdapterColsDifferentInputsOutputs<_, NUM_READ_OPS, NUM_WRITE_OPS> =
            local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Read fp
        self.memory_bridge
            .read(
                fp_addr::<AB::F>(),
                [local.from_state.fp],
                timestamp_pp(),
                &local.fp_read_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // If rs2 is an immediate value, constrain that:
        // 1. It's a 16-bit two's complement integer (stored in rs2_limbs[0] and rs2_limbs[1])
        // 2. It's properly sign-extended to NUM_LIMBS bytes (the upper limbs must match the sign bit)
        let rs2_limbs = ctx.reads[1].clone();
        let rs2_sign = rs2_limbs[2].clone();
        let rs2_imm = rs2_limbs[0].clone()
            + rs2_limbs[1].clone() * AB::Expr::from_usize(1 << RV32_CELL_BITS)
            + rs2_sign.clone() * AB::Expr::from_usize(1 << (2 * RV32_CELL_BITS));
        builder.assert_bool(local.rs2_as);
        let mut rs2_imm_when = builder.when(not(local.rs2_as));
        rs2_imm_when.assert_eq(local.rs2, rs2_imm);
        for limb in rs2_limbs.iter().skip(3) {
            rs2_imm_when.assert_eq(rs2_sign.clone(), limb.clone());
        }
        rs2_imm_when.assert_zero(
            rs2_sign.clone() * (AB::Expr::from_usize((1 << RV32_CELL_BITS) - 1) - rs2_sign),
        );
        self.bitwise_lookup_bus
            .send_range(rs2_limbs[0].clone(), rs2_limbs[1].clone())
            .eval(builder, ctx.instruction.is_valid.clone() - local.rs2_as);

        // rs1 reads: loop over register-sized chunks
        for r in 0..NUM_READ_OPS {
            let offset = r * RV32_REGISTER_NUM_LIMBS;
            let chunk: [AB::Expr; RV32_REGISTER_NUM_LIMBS] =
                std::array::from_fn(|i| ctx.reads[0][offset + i].clone());
            self.memory_bridge
                .read(
                    reg_addr(local.rs1_ptr + local.from_state.fp + AB::F::from_usize(offset)),
                    chunk,
                    timestamp_pp(),
                    &local.rs1_reads_aux[r],
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // This constraint ensures that the following memory reads only occur when `is_valid == 1`.
        builder
            .when(local.rs2_as)
            .assert_one(ctx.instruction.is_valid.clone());

        // rs2 reads: loop over register-sized chunks, enabled only when rs2_as=1
        for r in 0..NUM_READ_OPS {
            let offset = r * RV32_REGISTER_NUM_LIMBS;
            let chunk: [AB::Expr; RV32_REGISTER_NUM_LIMBS] =
                std::array::from_fn(|i| ctx.reads[1][offset + i].clone());
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        local.rs2_as,
                        local.rs2 + local.from_state.fp + AB::F::from_usize(offset),
                    ),
                    chunk,
                    timestamp_pp(),
                    &local.rs2_reads_aux[r],
                )
                .eval(builder, local.rs2_as);
        }

        // rd writes: loop over register-sized chunks (only NUM_WRITE_OPS chunks)
        for w in 0..NUM_WRITE_OPS {
            let offset = w * RV32_REGISTER_NUM_LIMBS;
            let chunk: [AB::Expr; RV32_REGISTER_NUM_LIMBS] =
                std::array::from_fn(|i| ctx.writes[0][offset + i].clone());
            self.memory_bridge
                .write(
                    reg_addr(local.rd_ptr + local.from_state.fp + AB::F::from_usize(offset)),
                    chunk,
                    timestamp_pp(),
                    &local.writes_aux[w],
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    local.rd_ptr.into(),
                    local.rs1_ptr.into(),
                    local.rs2.into(),
                    AB::Expr::from_u32(RV32_REGISTER_AS),
                    local.rs2_as.into(),
                ],
                local.from_state.into(),
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &BaseAluAdapterColsDifferentInputsOutputs<_, NUM_READ_OPS, NUM_WRITE_OPS> =
            local.borrow();
        cols.from_state.pc
    }
}

#[derive(Clone)]
pub struct BaseAluAdapterExecutorDifferentInputsOutputs<
    const NUM_LIMBS: usize,
    const NUM_READ_OPS: usize,
    const NUM_WRITE_OPS: usize,
> {
    /// Hack: This flag is used so that we fetch the frame pointer exactly once per instruction execution,
    ///       BEFORE the first read.
    has_fetched_fp: Arc<Mutex<bool>>,
}

impl<const NUM_LIMBS: usize, const NUM_READ_OPS: usize, const NUM_WRITE_OPS: usize> Default
    for BaseAluAdapterExecutorDifferentInputsOutputs<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS>
{
    fn default() -> Self {
        Self {
            has_fetched_fp: Arc::new(Mutex::new(false)),
        }
    }
}

#[derive(derive_new::new)]
pub struct BaseAluAdapterFillerDifferentInputsOutputs<
    const NUM_READ_OPS: usize,
    const NUM_WRITE_OPS: usize,
> {
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
}

// Intermediate type that should not be copied or cloned and should be directly written to
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct BaseAluAdapterRecordDifferentInputsOutputs<
    const NUM_READ_OPS: usize,
    const NUM_WRITE_OPS: usize,
> {
    pub from_pc: u32,
    pub fp: u32,
    pub from_timestamp: u32,

    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    /// Pointer if rs2 was a read, immediate value otherwise
    pub rs2: u32,
    /// 1 if rs2 was a read, 0 if an immediate
    pub rs2_as: u8,

    pub fp_read_aux: MemoryReadAuxRecord,
    pub rs1_reads_aux: [MemoryReadAuxRecord; NUM_READ_OPS],
    pub rs2_reads_aux: [MemoryReadAuxRecord; NUM_READ_OPS],
    pub writes_aux: [MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS>; NUM_WRITE_OPS],
}

impl<const NUM_LIMBS: usize, const NUM_READ_OPS: usize, const NUM_WRITE_OPS: usize>
    BaseAluAdapterExecutorDifferentInputsOutputs<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS>
{
    fn maybe_fetch_fp<F: PrimeField32>(
        &self,
        memory: &mut TracingMemory,
        record: &mut BaseAluAdapterRecordDifferentInputsOutputs<NUM_READ_OPS, NUM_WRITE_OPS>,
    ) {
        let mut has_fetched_fp = self
            .has_fetched_fp
            .lock()
            .expect("has_fetched_fp mutex poisoned");
        if !*has_fetched_fp {
            record.fp = tracing_read_fp::<F>(memory, &mut record.fp_read_aux.prev_timestamp);
            *has_fetched_fp = true;
        }
    }

    fn finalize_instruction(&self) {
        *self
            .has_fetched_fp
            .lock()
            .expect("has_fetched_fp mutex poisoned") = false;
    }
}

impl<F: PrimeField32, const NUM_LIMBS: usize, const NUM_READ_OPS: usize, const NUM_WRITE_OPS: usize>
    AdapterTraceExecutor<F>
    for BaseAluAdapterExecutorDifferentInputsOutputs<NUM_LIMBS, NUM_READ_OPS, NUM_WRITE_OPS>
{
    const WIDTH: usize =
        size_of::<BaseAluAdapterColsDifferentInputsOutputs<u8, NUM_READ_OPS, NUM_WRITE_OPS>>();
    type ReadData = [[u8; NUM_LIMBS]; 2];
    type WriteData = [[u8; NUM_LIMBS]; 1];
    type RecordMut<'a> =
        &'a mut BaseAluAdapterRecordDifferentInputsOutputs<NUM_READ_OPS, NUM_WRITE_OPS>;

    #[inline(always)]
    fn start(
        pc: u32,
        memory: &TracingMemory,
        record: &mut &mut BaseAluAdapterRecordDifferentInputsOutputs<NUM_READ_OPS, NUM_WRITE_OPS>,
    ) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    // @dev cannot get rid of double &mut due to trait
    #[inline(always)]
    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut &mut BaseAluAdapterRecordDifferentInputsOutputs<NUM_READ_OPS, NUM_WRITE_OPS>,
    ) -> Self::ReadData {
        let &Instruction { b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert!(
            e.as_canonical_u32() == RV32_REGISTER_AS || e.as_canonical_u32() == RV32_IMM_AS
        );

        self.maybe_fetch_fp::<F>(memory, record);

        record.rs1_ptr = b.as_canonical_u32();

        // Read rs1 in register-sized chunks
        let mut rs1 = [0u8; NUM_LIMBS];
        for r in 0..NUM_READ_OPS {
            let offset = r * RV32_REGISTER_NUM_LIMBS;
            let chunk: [u8; RV32_REGISTER_NUM_LIMBS] = tracing_read(
                memory,
                RV32_REGISTER_AS,
                record.rs1_ptr + record.fp + offset as u32,
                &mut record.rs1_reads_aux[r].prev_timestamp,
            );
            rs1[offset..offset + RV32_REGISTER_NUM_LIMBS].copy_from_slice(&chunk);
        }

        let rs2 = if e.as_canonical_u32() == RV32_REGISTER_AS {
            record.rs2_as = RV32_REGISTER_AS as u8;
            record.rs2 = c.as_canonical_u32();

            // Read rs2 in register-sized chunks
            let mut rs2 = [0u8; NUM_LIMBS];
            for r in 0..NUM_READ_OPS {
                let offset = r * RV32_REGISTER_NUM_LIMBS;
                let chunk: [u8; RV32_REGISTER_NUM_LIMBS] = tracing_read(
                    memory,
                    RV32_REGISTER_AS,
                    record.rs2 + record.fp + offset as u32,
                    &mut record.rs2_reads_aux[r].prev_timestamp,
                );
                rs2[offset..offset + RV32_REGISTER_NUM_LIMBS].copy_from_slice(&chunk);
            }
            rs2
        } else {
            record.rs2_as = RV32_IMM_AS as u8;

            // Immediate uses 1 timestamp, pad with extra increments for remaining NUM_READ_OPS-1
            let rs2 = tracing_read_imm::<NUM_LIMBS>(memory, c.as_canonical_u32(), &mut record.rs2);
            for _ in 1..NUM_READ_OPS {
                memory.increment_timestamp();
            }
            rs2
        };

        [rs1, rs2]
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut &mut BaseAluAdapterRecordDifferentInputsOutputs<NUM_READ_OPS, NUM_WRITE_OPS>,
    ) {
        let &Instruction { a, d, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);

        record.rd_ptr = a.as_canonical_u32();

        // Write rd in register-sized chunks (only NUM_WRITE_OPS chunks)
        for w in 0..NUM_WRITE_OPS {
            let offset = w * RV32_REGISTER_NUM_LIMBS;
            let chunk: [u8; RV32_REGISTER_NUM_LIMBS] = std::array::from_fn(|i| data[0][offset + i]);
            tracing_write(
                memory,
                RV32_REGISTER_AS,
                record.rd_ptr + record.fp + offset as u32,
                chunk,
                &mut record.writes_aux[w].prev_timestamp,
                &mut record.writes_aux[w].prev_data,
            );
        }

        self.finalize_instruction();
    }
}

impl<F: PrimeField32, const NUM_READ_OPS: usize, const NUM_WRITE_OPS: usize> AdapterTraceFiller<F>
    for BaseAluAdapterFillerDifferentInputsOutputs<NUM_READ_OPS, NUM_WRITE_OPS>
{
    const WIDTH: usize =
        size_of::<BaseAluAdapterColsDifferentInputsOutputs<u8, NUM_READ_OPS, NUM_WRITE_OPS>>();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY: the following is highly unsafe. We are going to cast `adapter_row` to a record
        // buffer, and then do an _overlapping_ write to the `adapter_row` as a row of field
        // elements. This requires:
        // - Cols struct should be repr(C) and we write in reverse order (to ensure non-overlapping)
        // - Do not overwrite any reference in `record` before it has already been used or moved
        // - alignment of `F` must be >= alignment of Record (AlignedBytesBorrow will panic
        //   otherwise)
        // - adapter_row contains a valid BaseAluAdapterRecord representation
        // - get_record_from_slice correctly interprets the bytes as BaseAluAdapterRecord
        let record: &BaseAluAdapterRecordDifferentInputsOutputs<NUM_READ_OPS, NUM_WRITE_OPS> =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut BaseAluAdapterColsDifferentInputsOutputs<
            F,
            NUM_READ_OPS,
            NUM_WRITE_OPS,
        > = adapter_row.borrow_mut();

        // We must assign in reverse
        // Total memory ops after fp read: NUM_READ_OPS (rs1) + NUM_READ_OPS (rs2) + NUM_WRITE_OPS (rd)
        let timestamp_delta: u32 = 2 * NUM_READ_OPS as u32 + NUM_WRITE_OPS as u32;
        let mut timestamp = record.from_timestamp + timestamp_delta;

        // Writes (reverse order)
        for w in (0..NUM_WRITE_OPS).rev() {
            adapter_row.writes_aux[w].set_prev_data(record.writes_aux[w].prev_data.map(F::from_u8));
            mem_helper.fill(
                record.writes_aux[w].prev_timestamp,
                timestamp,
                adapter_row.writes_aux[w].as_mut(),
            );
            timestamp -= 1;
        }

        // rs2 reads (reverse order)
        if record.rs2_as != 0 {
            for r in (0..NUM_READ_OPS).rev() {
                mem_helper.fill(
                    record.rs2_reads_aux[r].prev_timestamp,
                    timestamp,
                    adapter_row.rs2_reads_aux[r].as_mut(),
                );
                timestamp -= 1;
            }
        } else {
            for r in (0..NUM_READ_OPS).rev() {
                mem_helper.fill_zero(adapter_row.rs2_reads_aux[r].as_mut());
                timestamp -= 1;
            }
            let rs2_imm = record.rs2;
            let mask = (1 << RV32_CELL_BITS) - 1;
            self.bitwise_lookup_chip
                .request_range(rs2_imm & mask, (rs2_imm >> 8) & mask);
        }

        // rs1 reads (reverse order)
        for r in (0..NUM_READ_OPS).rev() {
            mem_helper.fill(
                record.rs1_reads_aux[r].prev_timestamp,
                timestamp,
                adapter_row.rs1_reads_aux[r].as_mut(),
            );
            timestamp -= 1;
        }

        // fp read
        mem_helper.fill(
            record.fp_read_aux.prev_timestamp,
            timestamp,
            adapter_row.fp_read_aux.as_mut(),
        );

        adapter_row.rs2_as = F::from_u8(record.rs2_as);
        adapter_row.rs2 = F::from_u32(record.rs2);
        adapter_row.rs1_ptr = F::from_u32(record.rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(record.rd_ptr);
        adapter_row.from_state.timestamp = F::from_u32(timestamp);
        adapter_row.from_state.fp = F::from_u32(record.fp);
        adapter_row.from_state.pc = F::from_u32(record.from_pc);
    }
}

// Convenience type aliases for the common case where read and write widths are the same.
pub type BaseAluAdapterCols<T, const NUM_OPS: usize> =
    BaseAluAdapterColsDifferentInputsOutputs<T, NUM_OPS, NUM_OPS>;
pub type BaseAluAdapterAir<const NUM_LIMBS: usize, const NUM_OPS: usize> =
    BaseAluAdapterAirDifferentInputsOutputs<NUM_LIMBS, NUM_OPS, NUM_OPS>;
pub type BaseAluAdapterExecutor<const NUM_LIMBS: usize, const NUM_OPS: usize> =
    BaseAluAdapterExecutorDifferentInputsOutputs<NUM_LIMBS, NUM_OPS, NUM_OPS>;
pub type BaseAluAdapterFiller<const NUM_OPS: usize> =
    BaseAluAdapterFillerDifferentInputsOutputs<NUM_OPS, NUM_OPS>;
pub type BaseAluAdapterRecord<const NUM_OPS: usize> =
    BaseAluAdapterRecordDifferentInputsOutputs<NUM_OPS, NUM_OPS>;

// 32-bit type aliases
pub type Rv32BaseAluAdapterCols<T> = BaseAluAdapterCols<T, W32_REG_OPS>;
pub type Rv32BaseAluAdapterAir = BaseAluAdapterAir<RV32_REGISTER_NUM_LIMBS, W32_REG_OPS>;
pub type Rv32BaseAluAdapterRecord = BaseAluAdapterRecord<W32_REG_OPS>;
pub type Rv32BaseAluAdapterExecutor = BaseAluAdapterExecutor<RV32_REGISTER_NUM_LIMBS, W32_REG_OPS>;
pub type Rv32BaseAluAdapterFiller = BaseAluAdapterFiller<W32_REG_OPS>;
