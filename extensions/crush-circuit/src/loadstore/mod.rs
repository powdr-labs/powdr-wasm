use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_rv32im_circuit::{LoadStoreCoreAir, LoadStoreFiller};

use super::adapters::RV32_REGISTER_NUM_LIMBS;
use crate::{
    adapters::{LoadStoreAdapterAir, LoadStoreAdapterFiller},
    loadstore::execution::LoadStoreExecutor,
};

pub mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::LoadStoreChipGpu;

pub type LoadStoreAir =
    VmAirWrapper<LoadStoreAdapterAir, LoadStoreCoreAir<RV32_REGISTER_NUM_LIMBS>>;
pub type LoadStore32Executor = LoadStoreExecutor<RV32_REGISTER_NUM_LIMBS>;
pub type LoadStoreChip<F> = VmChipWrapper<F, LoadStoreFiller<LoadStoreAdapterFiller>>;
