use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{MultiplicationCoreAir, MultiplicationFiller};

use super::adapters::{
    BaseAluAdapter32Air, BaseAluAdapter32Filler, BaseAluAdapterAir, BaseAluAdapterFiller,
    RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, W32_REG_OPS, W64_NUM_LIMBS, W64_REG_OPS,
};

mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::{Mul64ChipGpu, Multiplication32ChipGpu};

pub use execution::MultiplicationExecutor;

// 32-bit type aliases
pub type Multiplication32Air = VmAirWrapper<
    BaseAluAdapter32Air,
    MultiplicationCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Multiplication32Executor = MultiplicationExecutor<RV32_REGISTER_NUM_LIMBS, W32_REG_OPS>;
pub type Multiplication32Chip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<BaseAluAdapter32Filler, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

// 64-bit type aliases
pub type Mul64Air = VmAirWrapper<
    BaseAluAdapterAir<W64_NUM_LIMBS, W64_REG_OPS>,
    MultiplicationCoreAir<W64_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Mul64Executor = MultiplicationExecutor<W64_NUM_LIMBS, W64_REG_OPS>;
pub type Mul64Chip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<BaseAluAdapterFiller<W64_REG_OPS>, W64_NUM_LIMBS, RV32_CELL_BITS>,
>;
