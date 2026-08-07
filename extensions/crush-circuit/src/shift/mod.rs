use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{ShiftCoreAir, ShiftFiller};

use super::adapters::{
    BaseAluAdapter32Air, BaseAluAdapter32Filler, BaseAluAdapterAir, BaseAluAdapterFiller,
    RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS, W32_REG_OPS, W64_NUM_LIMBS, W64_REG_OPS,
};

mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::{Shift32ChipGpu, Shift64ChipGpu};

pub use execution::ShiftExecutor;

// 32-bit type aliases
pub type Shift32Air =
    VmAirWrapper<BaseAluAdapter32Air, ShiftCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type Shift32Executor = ShiftExecutor<RV32_REGISTER_NUM_LIMBS, W32_REG_OPS>;
pub type Shift32Chip<F> =
    VmChipWrapper<F, ShiftFiller<BaseAluAdapter32Filler, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;

// 64-bit type aliases
pub type Shift64Air = VmAirWrapper<
    BaseAluAdapterAir<W64_NUM_LIMBS, W64_REG_OPS>,
    ShiftCoreAir<W64_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Shift64Executor = ShiftExecutor<W64_NUM_LIMBS, W64_REG_OPS>;
pub type Shift64Chip<F> =
    VmChipWrapper<F, ShiftFiller<BaseAluAdapterFiller<W64_REG_OPS>, W64_NUM_LIMBS, RV32_CELL_BITS>>;
