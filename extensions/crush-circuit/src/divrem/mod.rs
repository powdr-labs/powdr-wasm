use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{DivRemCoreAir, DivRemFiller};

use super::adapters::{
    BaseAluAdapter32Air, BaseAluAdapter32Filler, BaseAluAdapterAir, BaseAluAdapterFiller,
    RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, W32_REG_OPS, W64_NUM_LIMBS, W64_REG_OPS,
};

mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::{DivRem32ChipGpu, DivRem64ChipGpu};

pub use execution::DivRemExecutor;

// 32-bit type aliases
pub type DivRem32Air =
    VmAirWrapper<BaseAluAdapter32Air, DivRemCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type DivRem32Executor = DivRemExecutor<RV32_REGISTER_NUM_LIMBS, W32_REG_OPS>;
pub type DivRem32Chip<F> =
    VmChipWrapper<F, DivRemFiller<BaseAluAdapter32Filler, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;

// 64-bit type aliases
pub type DivRem64Air = VmAirWrapper<
    BaseAluAdapterAir<W64_NUM_LIMBS, W64_REG_OPS>,
    DivRemCoreAir<W64_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type DivRem64Executor = DivRemExecutor<W64_NUM_LIMBS, W64_REG_OPS>;
pub type DivRem64Chip<F> = VmChipWrapper<
    F,
    DivRemFiller<BaseAluAdapterFiller<W64_REG_OPS>, W64_NUM_LIMBS, RV32_CELL_BITS>,
>;
