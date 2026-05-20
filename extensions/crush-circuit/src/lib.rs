#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]
use crate::memory_config::memory_config_with_fp;
use keccak_circuit::{Keccak256, Keccak256CpuProverExt, Keccak256Executor};
use openvm_circuit::{
    arch::{
        AirInventory, ChipInventoryError, InitFileGenerator, MatrixRecordArena, SystemConfig,
        VmBuilder, VmChipComplex, VmProverExtension,
    },
    system::{SystemChipInventory, SystemCpuBuilder, SystemExecutor},
};
use openvm_circuit_derive::{Executor, MeteredExecutor, VmConfig};
use openvm_sdk::config::TranspilerConfig;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use openvm_transpiler::transpiler::Transpiler;
use powdr_openvm::{SpecializedExecutor, isa::OpenVmISA};
use serde::{Deserialize, Serialize};

pub mod execution;

pub mod adapters;
mod base_alu;
mod call;
mod const32;
mod divrem;
mod eq;
mod jump;
mod less_than;
mod load_sign_extend;
mod loadstore;
mod mul;
mod shift;
mod utils;

pub use base_alu::*;
pub use call::*;
pub use const32::*;
pub use divrem::*;
pub use eq::*;
pub use jump::*;
pub use less_than::*;
pub use load_sign_extend::*;
pub use loadstore::*;
pub use mul::*;
pub use shift::*;

mod hintstore;
pub use hintstore::*;

mod extension;
pub use extension::*;

pub mod memory_config;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_circuit::arch::DenseRecordArena;
        use openvm_circuit::system::cuda::{extensions::SystemGpuBuilder, SystemChipInventoryGPU};
        use openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
        use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
        pub(crate) mod cuda_abi;
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        pub use self::CrushGpuBuilder as CrushBuilder;
    } else {
        pub use self::CrushCpuBuilder as CrushBuilder;
    }
}

#[cfg(any(test, feature = "test-utils"))]
mod test_utils;

// Config for a VM with base extension and IO extension
#[derive(Clone, Debug, derive_new::new, VmConfig, Serialize, Deserialize)]
pub struct CrushConfig {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub base: Crush,
    #[extension(executor = "Keccak256Executor")]
    pub keccak: Option<Keccak256>,
}

// This seems trivial but it's tricky to put into powdr-openvm because of some From implementation issues.
impl<F: PrimeField32, ISA: OpenVmISA<Executor<F> = CrushConfigExecutor<F>>>
    From<CrushConfigExecutor<F>> for SpecializedExecutor<F, ISA>
{
    fn from(value: CrushConfigExecutor<F>) -> Self {
        Self::OriginalExecutor(value)
    }
}

// Default implementation uses no init file
impl InitFileGenerator for CrushConfig {}

impl Default for CrushConfig {
    fn default() -> Self {
        let system = system_config();
        Self {
            system,
            base: Default::default(),
            keccak: None,
        }
    }
}

impl CrushConfig {
    pub fn with_public_values(public_values: usize) -> Self {
        let system = system_config().with_public_values(public_values);
        Self {
            system,
            base: Default::default(),
            keccak: None,
        }
    }

    pub fn with_public_values_and_segment_len(public_values: usize, segment_len: usize) -> Self {
        let system = system_config()
            .with_public_values(public_values)
            .with_max_segment_len(segment_len);
        Self {
            system,
            base: Default::default(),
            keccak: None,
        }
    }

    pub fn with_keccak(mut self) -> Self {
        self.keccak = Some(Keccak256);
        self
    }
}

impl<F: PrimeField32> TranspilerConfig<F> for CrushConfig {
    fn transpiler(&self) -> Transpiler<F> {
        // Crush programs are lowered directly to OpenVM instructions.
        Transpiler::default()
    }
}

#[derive(Clone, Default)]
pub struct CrushCpuBuilder;

impl<E, SC> VmBuilder<E> for CrushCpuBuilder
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: PrimeField32,
{
    type VmConfig = CrushConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &CrushConfig,
        circuit: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemCpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&CrushCpuProverExt, &config.base, inventory)?;
        if let Some(ref keccak) = config.keccak {
            VmProverExtension::<E, _, _>::extend_prover(&Keccak256CpuProverExt, keccak, inventory)?;
        }
        Ok(chip_complex)
    }
}

pub fn system_config() -> SystemConfig {
    SystemConfig::default_from_memory(memory_config_with_fp())
}

#[cfg(feature = "cuda")]
#[derive(Clone, Default)]
pub struct CrushGpuBuilder;

#[cfg(feature = "cuda")]
impl VmBuilder<GpuBabyBearPoseidon2Engine> for CrushGpuBuilder {
    type VmConfig = CrushConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &CrushConfig,
        circuit: AirInventory<BabyBearPoseidon2Config>,
    ) -> Result<
        VmChipComplex<
            BabyBearPoseidon2Config,
            Self::RecordArena,
            GpuBackend,
            Self::SystemChipInventory,
        >,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<GpuBabyBearPoseidon2Engine>::create_chip_complex(
            &SystemGpuBuilder,
            &config.system,
            circuit,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &CrushGpuProverExt,
            &config.base,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
