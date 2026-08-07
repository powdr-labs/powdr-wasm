use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};

use crate::{
    adapters::call::{CallAdapterAir, CallAdapterExecutor},
    call::tracegen::CallFiller,
};

mod core;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod execution;
pub mod tracegen;

pub use self::core::{CallCoreAir, CallCoreCols, CallCoreRecord};
pub use execution::CallExecutor;

#[cfg(feature = "cuda")]
pub use cuda::CallChipGpu;

pub type Call32Executor = CallExecutor<CallAdapterExecutor>;
pub type CallAir = VmAirWrapper<CallAdapterAir, CallCoreAir>;
pub type CallChip<F> = VmChipWrapper<F, CallFiller>;
