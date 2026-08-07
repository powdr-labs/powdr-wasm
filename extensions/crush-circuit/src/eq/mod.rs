use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use super::adapters::{
    BaseAluAdapter32Air, BaseAluAdapter32Filler, BaseAluAdapterAirDifferentInputsOutputs,
    BaseAluAdapterFillerDifferentInputsOutputs, RV32_REGISTER_NUM_LIMBS, W32_REG_OPS,
    W64_NUM_LIMBS, W64_REG_OPS,
};

pub(crate) mod core;
mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::{Eq32ChipGpu, Eq64ChipGpu};

pub use self::core::{EqCoreAir, EqFiller};
pub use execution::EqExecutor;

// 32-bit type aliases
pub type Eq32Air = VmAirWrapper<BaseAluAdapter32Air, EqCoreAir<RV32_REGISTER_NUM_LIMBS>>;
pub type Eq32Executor = EqExecutor<RV32_REGISTER_NUM_LIMBS, W32_REG_OPS>;
pub type Eq32Chip<F> = VmChipWrapper<F, EqFiller<BaseAluAdapter32Filler, RV32_REGISTER_NUM_LIMBS>>;

// 64-bit type aliases
// Reads 64-bit operands (W64_REG_OPS=2 reads per operand), but comparison
// results are i32 so only one 32-bit register write (W32_REG_OPS=1).
pub type Eq64Air = VmAirWrapper<
    BaseAluAdapterAirDifferentInputsOutputs<W64_NUM_LIMBS, W64_REG_OPS, W32_REG_OPS>,
    EqCoreAir<W64_NUM_LIMBS>,
>;
pub type Eq64Executor = EqExecutor<W64_NUM_LIMBS, W64_REG_OPS>;
pub type Eq64Chip<F> = VmChipWrapper<
    F,
    EqFiller<BaseAluAdapterFillerDifferentInputsOutputs<W64_REG_OPS, W32_REG_OPS>, W64_NUM_LIMBS>,
>;
