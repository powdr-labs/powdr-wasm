use crate::adapters::{VecHeapAdapterAir, VecHeapAdapterFiller};
use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionFiller};

use crate::algebra::FieldExprVecHeapExecutor;

mod addsub;
pub use addsub::*;
mod muldiv;
pub use muldiv::*;

pub type Fp2Air<const BLOCKS: usize, const BLOCK_SIZE: usize> = VmAirWrapper<
    VecHeapAdapterAir<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

pub type Fp2Executor<const BLOCKS: usize, const BLOCK_SIZE: usize> =
    FieldExprVecHeapExecutor<BLOCKS, BLOCK_SIZE, true>;

pub type Fp2Chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> = VmChipWrapper<
    F,
    FieldExpressionFiller<VecHeapAdapterFiller<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>,
>;
