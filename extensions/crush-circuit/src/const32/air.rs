use std::borrow::Borrow;

use crate::adapters::{fp_addr, fp_block, reg_addr};
use crate::execution::ExecutionState;
use openvm_circuit::arch::{ExecutionBridge, ExecutionState as OvmExecutionState};
use openvm_circuit::system::memory::offline_checker::{
    MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols,
};
use openvm_circuit_primitives::{AlignedBorrow, bitwise_op_lookup::BitwiseOperationLookupBus};
use openvm_instructions::program::DEFAULT_PC_STEP;
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_stark_backend::interaction::InteractionBuilder;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_backend::{
    BaseAirWithPublicValues, ColumnsAir, PartitionedBaseAir,
    p3_air::{Air, BaseAir},
    p3_matrix::Matrix,
};
use struct_reflection::{StructReflection, StructReflectionHelper};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct Const32AdapterAirCol<T, const NUM_LIMBS: usize> {
    pub is_valid: T,
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub imm_limbs: [T; NUM_LIMBS],
    pub fp_read_aux: MemoryReadAuxCols<T>,
    pub write_aux: MemoryWriteAuxCols<T, NUM_LIMBS>,
}

#[derive(derive_new::new)]
pub struct Const32AdapterAir<const NUM_LIMBS: usize> {
    pub bus: BitwiseOperationLookupBus,
    pub offset: usize,
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F, const NUM_LIMBS: usize> BaseAir<F> for Const32AdapterAir<NUM_LIMBS> {
    fn width(&self) -> usize {
        Const32AdapterAirCol::<F, NUM_LIMBS>::width()
    }
}

impl<F, const NUM_LIMBS: usize> BaseAirWithPublicValues<F> for Const32AdapterAir<NUM_LIMBS> {
    fn num_public_values(&self) -> usize {
        0
    }
}

impl<F, const NUM_LIMBS: usize> PartitionedBaseAir<F> for Const32AdapterAir<NUM_LIMBS> {}

impl<F, const NUM_LIMBS: usize> ColumnsAir<F> for Const32AdapterAir<NUM_LIMBS> {
    fn columns(&self) -> Option<Vec<String>> {
        Const32AdapterAirCol::<F, NUM_LIMBS>::struct_reflection()
    }
}

impl<AB, const NUM_LIMBS: usize> Air<AB> for Const32AdapterAir<NUM_LIMBS>
where
    AB: InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("row_slice(0) should exist");
        let cols: &Const32AdapterAirCol<AB::Var, NUM_LIMBS> = (*local).borrow();

        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Read fp
        self.memory_bridge
            .read(
                fp_addr::<AB::F>(),
                fp_block::<AB::Expr>(cols.from_state.fp.into()),
                timestamp_pp(),
                &cols.fp_read_aux,
            )
            .eval(builder, cols.is_valid);

        // Write imm_limbs to register at rd_ptr + fp
        self.memory_bridge
            .write(
                reg_addr(cols.rd_ptr + cols.from_state.fp),
                cols.imm_limbs,
                timestamp_pp(),
                &cols.write_aux,
            )
            .eval(builder, cols.is_valid);

        // Range-check imm_limbs via bitwise lookup bus
        self.bus
            .send_range(cols.imm_limbs[0], cols.imm_limbs[1])
            .eval(builder, cols.is_valid);
        self.bus
            .send_range(cols.imm_limbs[2], cols.imm_limbs[3])
            .eval(builder, cols.is_valid);

        // Reconstruct instruction operands b (imm_lo) and c (imm_hi) from limbs
        let cell_factor = AB::F::from_u32(1 << RV32_CELL_BITS);
        let imm_lo: AB::Expr = cols.imm_limbs[0] + cols.imm_limbs[1] * cell_factor;
        let imm_hi: AB::Expr = cols.imm_limbs[2] + cols.imm_limbs[3] * cell_factor;

        // Execution bridge: verify instruction and advance PC
        self.execution_bridge
            .execute(
                AB::F::from_usize(self.offset),
                [
                    cols.rd_ptr.into(),
                    imm_lo,
                    imm_hi,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ONE,
                    AB::Expr::ZERO,
                ],
                cols.from_state.into(),
                OvmExecutionState {
                    pc: cols.from_state.pc + AB::F::from_u32(DEFAULT_PC_STEP),
                    timestamp: timestamp + AB::F::from_usize(timestamp_delta),
                },
            )
            .eval(builder, cols.is_valid);

        builder.assert_bool(cols.is_valid);
    }
}
