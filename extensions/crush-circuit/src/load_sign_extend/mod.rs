use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{LoadSignExtendCoreAir, LoadSignExtendFiller};

use super::adapters::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use crate::{
    adapters::{LoadStoreAdapterAir, LoadStoreAdapterFiller},
    load_sign_extend::execution::LoadSignExtendExecutor,
};

pub mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::LoadSignExtendChipGpu;

pub type LoadSignExtendAir = VmAirWrapper<
    LoadStoreAdapterAir,
    LoadSignExtendCoreAir<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type LoadSignExtend32Executor = LoadSignExtendExecutor<RV32_REGISTER_NUM_LIMBS>;
pub type LoadSignExtendChip<F> = VmChipWrapper<F, LoadSignExtendFiller<LoadStoreAdapterFiller>>;
