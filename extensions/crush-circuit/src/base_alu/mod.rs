use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{BaseAluCoreAir, BaseAluFiller};

use super::adapters::{
    BaseAluAdapter32Air, BaseAluAdapter32Filler, BaseAluAdapterAir, BaseAluAdapterFiller,
    RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, W32_REG_OPS, W64_NUM_LIMBS, W64_REG_OPS,
};

mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::{BaseAlu32ChipGpu, BaseAlu64ChipGpu};

pub use execution::BaseAluExecutor;

// 32-bit type aliases
pub type BaseAlu32Air =
    VmAirWrapper<BaseAluAdapter32Air, BaseAluCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type BaseAlu32Executor = BaseAluExecutor<RV32_REGISTER_NUM_LIMBS, W32_REG_OPS>;
pub type BaseAlu32Chip<F> = VmChipWrapper<
    F,
    BaseAluFiller<BaseAluAdapter32Filler, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

// 64-bit type aliases
pub type BaseAlu64Air = VmAirWrapper<
    BaseAluAdapterAir<W64_NUM_LIMBS, W64_REG_OPS>,
    BaseAluCoreAir<W64_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type BaseAlu64Executor = BaseAluExecutor<W64_NUM_LIMBS, W64_REG_OPS>;
pub type BaseAlu64Chip<F> = VmChipWrapper<
    F,
    BaseAluFiller<BaseAluAdapterFiller<W64_REG_OPS>, W64_NUM_LIMBS, RV32_CELL_BITS>,
>;
