use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{
    Chip, bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{GpuBackend, base::DeviceMatrix, prelude::F};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_rv32im_circuit::ShiftCoreCols;
use openvm_stark_backend::prover::AirProvingContext;

use crate::{
    adapters::{
        BaseAluAdapter32Cols, BaseAluAdapter32Record, BaseAluAdapterCols, BaseAluAdapterRecord,
        RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, W64_NUM_LIMBS, W64_REG_OPS,
    },
    cuda_abi::{shift_cuda, shift64_cuda},
};

use openvm_rv32im_circuit::ShiftCoreRecord;

#[derive(new)]
pub struct Shift32ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Shift32ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            BaseAluAdapter32Record,
            ShiftCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = ShiftCoreCols::<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>::width()
            + BaseAluAdapter32Cols::<F>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            shift_cuda::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.range_checker.count.len(),
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

#[derive(new)]
pub struct Shift64ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Shift64ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            BaseAluAdapterRecord<W64_REG_OPS>,
            ShiftCoreRecord<W64_NUM_LIMBS, RV32_CELL_BITS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = ShiftCoreCols::<F, W64_NUM_LIMBS, RV32_CELL_BITS>::width()
            + BaseAluAdapterCols::<F, W64_REG_OPS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            shift64_cuda::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.range_checker.count.len(),
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}
