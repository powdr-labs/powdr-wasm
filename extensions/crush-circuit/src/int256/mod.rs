// The #[create_handler] macro generates unsafe fn bodies that call unsafe functions without
// inner unsafe blocks. Upstream's bigint crate is edition 2021, where that is allowed;
// crush-circuit is edition 2024, where it warns. Scoped to this module.
#![allow(unsafe_op_in_unsafe_fn)]

//! Int256 support, ported from OpenVM's `openvm-bigint-circuit` (v2.0.0-beta.2).
//!
//! The core AIRs, executors and fillers are reused from `openvm-rv32im-circuit` exactly as
//! upstream does -- these are the same cores crush already reuses for its 32- and 64-bit
//! chips. Two things are crush-specific: the heap adapters (forked in `crate::adapters` so
//! register reads are frame-pointer relative) and the interpreters in this module, which
//! read registers directly on the e1/e2 fast paths and so need the frame pointer too.

use openvm_circuit::arch::{DEFAULT_BLOCK_SIZE, VmAirWrapper, VmChipWrapper};
use openvm_circuit_derive::PreflightExecutor;
// The VecToFlat wrappers are generic over their inner adapter and perform no register
// access, so they come from upstream unchanged. The inner heap adapters are crush's
// FP-aware forks.
use openvm_rv32_adapters::{
    VecToFlatAluAdapterAir, VecToFlatAluAdapterExecutor, VecToFlatBranchAdapterAir,
    VecToFlatBranchAdapterExecutor,
};

use crate::adapters::{
    VecHeapAdapterAir, VecHeapAdapterExecutor, VecHeapAdapterFiller, VecHeapBranchAdapterAir,
    VecHeapBranchAdapterExecutor, VecHeapBranchAdapterFiller,
};
// Core AIRs, executors and fillers are reused from OpenVM unchanged: only the register
// addressing differs under crush, and that lives in the adapter and the interpreter.
use openvm_rv32im_circuit::{
    BaseAluCoreAir, BaseAluExecutor, BaseAluFiller, BranchEqualCoreAir, BranchEqualExecutor,
    BranchEqualFiller, BranchLessThanCoreAir, BranchLessThanExecutor, BranchLessThanFiller,
    LessThanCoreAir, LessThanExecutor, LessThanFiller, MultiplicationCoreAir,
    MultiplicationExecutor, MultiplicationFiller, ShiftCoreAir, ShiftExecutor, ShiftFiller,
    adapters::{INT256_NUM_LIMBS, RV32_CELL_BITS},
};

mod extension;
pub use extension::*;

mod base_alu;
mod branch_eq;
mod branch_lt;
pub(crate) mod common;
mod less_than;
mod mult;
mod shift;

/// Number of blocks for INT256 operations (INT256_NUM_LIMBS / DEFAULT_BLOCK_SIZE)
pub const INT256_NUM_BLOCKS: usize = INT256_NUM_LIMBS / DEFAULT_BLOCK_SIZE;

/// Type alias for the ALU adapter AIR wrapper
type AluAdapterAir = VecToFlatAluAdapterAir<
    VecHeapAdapterAir<
        2,
        INT256_NUM_BLOCKS,
        INT256_NUM_BLOCKS,
        DEFAULT_BLOCK_SIZE,
        DEFAULT_BLOCK_SIZE,
    >,
    2,
    INT256_NUM_BLOCKS,
    INT256_NUM_BLOCKS,
    DEFAULT_BLOCK_SIZE,
    INT256_NUM_LIMBS,
    INT256_NUM_LIMBS,
>;

/// Type alias for the ALU adapter executor wrapper
type AluAdapterExecutor = VecToFlatAluAdapterExecutor<
    VecHeapAdapterExecutor<
        2,
        INT256_NUM_BLOCKS,
        INT256_NUM_BLOCKS,
        DEFAULT_BLOCK_SIZE,
        DEFAULT_BLOCK_SIZE,
    >,
    2,
    INT256_NUM_BLOCKS,
    INT256_NUM_BLOCKS,
    DEFAULT_BLOCK_SIZE,
    INT256_NUM_LIMBS,
    INT256_NUM_LIMBS,
>;

/// Type alias for the Branch adapter AIR wrapper
type BranchAdapterAir = VecToFlatBranchAdapterAir<
    VecHeapBranchAdapterAir<2, INT256_NUM_BLOCKS, DEFAULT_BLOCK_SIZE>,
    2,
    INT256_NUM_BLOCKS,
    DEFAULT_BLOCK_SIZE,
    INT256_NUM_LIMBS,
>;

/// Type alias for the Branch adapter executor wrapper
type BranchAdapterExecutor = VecToFlatBranchAdapterExecutor<
    VecHeapBranchAdapterExecutor<2, INT256_NUM_BLOCKS, DEFAULT_BLOCK_SIZE>,
    2,
    INT256_NUM_BLOCKS,
    DEFAULT_BLOCK_SIZE,
    INT256_NUM_LIMBS,
>;

/// BaseAlu256
pub type Rv32BaseAlu256Air =
    VmAirWrapper<AluAdapterAir, BaseAluCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32BaseAlu256Executor(
    BaseAluExecutor<AluAdapterExecutor, INT256_NUM_LIMBS, RV32_CELL_BITS>,
);
pub type Rv32BaseAlu256Chip<F> = VmChipWrapper<
    F,
    BaseAluFiller<
        VecHeapAdapterFiller<
            2,
            INT256_NUM_BLOCKS,
            INT256_NUM_BLOCKS,
            DEFAULT_BLOCK_SIZE,
            DEFAULT_BLOCK_SIZE,
        >,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// LessThan256
pub type Rv32LessThan256Air =
    VmAirWrapper<AluAdapterAir, LessThanCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32LessThan256Executor(
    LessThanExecutor<AluAdapterExecutor, INT256_NUM_LIMBS, RV32_CELL_BITS>,
);
pub type Rv32LessThan256Chip<F> = VmChipWrapper<
    F,
    LessThanFiller<
        VecHeapAdapterFiller<
            2,
            INT256_NUM_BLOCKS,
            INT256_NUM_BLOCKS,
            DEFAULT_BLOCK_SIZE,
            DEFAULT_BLOCK_SIZE,
        >,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// Multiplication256
pub type Rv32Multiplication256Air =
    VmAirWrapper<AluAdapterAir, MultiplicationCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32Multiplication256Executor(
    MultiplicationExecutor<AluAdapterExecutor, INT256_NUM_LIMBS, RV32_CELL_BITS>,
);
pub type Rv32Multiplication256Chip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<
        VecHeapAdapterFiller<
            2,
            INT256_NUM_BLOCKS,
            INT256_NUM_BLOCKS,
            DEFAULT_BLOCK_SIZE,
            DEFAULT_BLOCK_SIZE,
        >,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// Shift256
pub type Rv32Shift256Air =
    VmAirWrapper<AluAdapterAir, ShiftCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32Shift256Executor(
    ShiftExecutor<AluAdapterExecutor, INT256_NUM_LIMBS, RV32_CELL_BITS>,
);
pub type Rv32Shift256Chip<F> = VmChipWrapper<
    F,
    ShiftFiller<
        VecHeapAdapterFiller<
            2,
            INT256_NUM_BLOCKS,
            INT256_NUM_BLOCKS,
            DEFAULT_BLOCK_SIZE,
            DEFAULT_BLOCK_SIZE,
        >,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// BranchEqual256
pub type Rv32BranchEqual256Air =
    VmAirWrapper<BranchAdapterAir, BranchEqualCoreAir<INT256_NUM_LIMBS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32BranchEqual256Executor(BranchEqualExecutor<BranchAdapterExecutor, INT256_NUM_LIMBS>);
pub type Rv32BranchEqual256Chip<F> = VmChipWrapper<
    F,
    BranchEqualFiller<
        VecHeapBranchAdapterFiller<2, INT256_NUM_BLOCKS, DEFAULT_BLOCK_SIZE>,
        INT256_NUM_LIMBS,
    >,
>;

/// BranchLessThan256
pub type Rv32BranchLessThan256Air =
    VmAirWrapper<BranchAdapterAir, BranchLessThanCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>>;
#[derive(Clone, PreflightExecutor)]
pub struct Rv32BranchLessThan256Executor(
    BranchLessThanExecutor<BranchAdapterExecutor, INT256_NUM_LIMBS, RV32_CELL_BITS>,
);
pub type Rv32BranchLessThan256Chip<F> = VmChipWrapper<
    F,
    BranchLessThanFiller<
        VecHeapBranchAdapterFiller<2, INT256_NUM_BLOCKS, DEFAULT_BLOCK_SIZE>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;
