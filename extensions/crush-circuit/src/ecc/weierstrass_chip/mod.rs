mod add_ne;
mod curves;
mod double;
mod preflight;

pub use add_ne::*;
pub use curves::CurveType;
pub use double::*;

use crate::adapters::{VecHeapAdapterAir, VecHeapAdapterFiller};
use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionFiller};

pub type WeierstrassAir<const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize> =
    VmAirWrapper<
        VecHeapAdapterAir<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreAir,
    >;

pub type WeierstrassChip<F, const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize> =
    VmChipWrapper<
        F,
        FieldExpressionFiller<
            VecHeapAdapterFiller<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >,
    >;
