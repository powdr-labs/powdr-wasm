use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{LessThanCoreAir, LessThanFiller};

use super::adapters::{
    BaseAluAdapter32Air, BaseAluAdapter32Filler, BaseAluAdapterAirDifferentInputsOutputs,
    BaseAluAdapterFillerDifferentInputsOutputs, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    W32_REG_OPS, W64_NUM_LIMBS, W64_REG_OPS,
};

mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::{LessThan32ChipGpu, LessThan64ChipGpu};

pub use execution::LessThanExecutor;

// 32-bit type aliases
pub type LessThan32Air =
    VmAirWrapper<BaseAluAdapter32Air, LessThanCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;
pub type LessThan32Executor = LessThanExecutor<RV32_REGISTER_NUM_LIMBS, W32_REG_OPS, W32_REG_OPS>;
pub type LessThan32Chip<F> = VmChipWrapper<
    F,
    LessThanFiller<BaseAluAdapter32Filler, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;

// 64-bit type aliases
// Reads 64-bit operands (W64_REG_OPS=2 reads per operand), but comparison
// results are i32 so only one 32-bit register write (W32_REG_OPS=1).
pub type LessThan64Air = VmAirWrapper<
    BaseAluAdapterAirDifferentInputsOutputs<W64_NUM_LIMBS, W64_REG_OPS, W32_REG_OPS>,
    LessThanCoreAir<W64_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type LessThan64Executor = LessThanExecutor<W64_NUM_LIMBS, W64_REG_OPS, W32_REG_OPS>;
pub type LessThan64Chip<F> = VmChipWrapper<
    F,
    LessThanFiller<
        BaseAluAdapterFillerDifferentInputsOutputs<W64_REG_OPS, W32_REG_OPS>,
        W64_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;
