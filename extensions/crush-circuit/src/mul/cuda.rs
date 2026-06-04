use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{
    Chip, bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
    var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{GpuBackend, base::DeviceMatrix, prelude::F};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_rv32im_circuit::{MultiplicationCoreCols, MultiplicationCoreRecord};
use openvm_stark_backend::prover::AirProvingContext;

use crate::{
    adapters::{
        BaseAluAdapterCols, BaseAluAdapterRecord, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
        Rv32BaseAluAdapterCols, Rv32BaseAluAdapterRecord, W64_NUM_LIMBS, W64_REG_OPS,
    },
    cuda_abi::{UInt2, mul_cuda, mul64_cuda},
};

#[derive(new)]
pub struct Rv32MultiplicationChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32MultiplicationChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv32BaseAluAdapterRecord,
            MultiplicationCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width =
            MultiplicationCoreCols::<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>::width()
                + Rv32BaseAluAdapterCols::<F>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let tuple_checker_sizes = self.range_tuple_checker.sizes;
        let tuple_checker_sizes = UInt2::new(tuple_checker_sizes[0], tuple_checker_sizes[1]);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            mul_cuda::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.range_checker.count.len(),
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                &self.range_tuple_checker.count,
                tuple_checker_sizes,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}

#[derive(new)]
pub struct Mul64ChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Mul64ChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            BaseAluAdapterRecord<W64_REG_OPS>,
            MultiplicationCoreRecord<W64_NUM_LIMBS, RV32_CELL_BITS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = MultiplicationCoreCols::<F, W64_NUM_LIMBS, RV32_CELL_BITS>::width()
            + BaseAluAdapterCols::<F, W64_REG_OPS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let tuple_checker_sizes = self.range_tuple_checker.sizes;
        let tuple_checker_sizes = UInt2::new(tuple_checker_sizes[0], tuple_checker_sizes[1]);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            mul64_cuda::tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.range_checker.count.len(),
                &self.bitwise_lookup.count,
                RV32_CELL_BITS,
                &self.range_tuple_checker.count,
                tuple_checker_sizes,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(d_trace)
    }
}
