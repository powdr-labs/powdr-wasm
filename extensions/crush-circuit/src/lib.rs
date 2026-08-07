#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]
use crate::algebra::{
    AlgebraCpuProverExt, Fp2Extension, Fp2ExtensionExecutor, ModularExtension,
    ModularExtensionExecutor,
};
use crate::ecc::{EccCpuProverExt, WeierstrassExtension, WeierstrassExtensionExecutor};
use crate::int256::{Int256, Int256CpuProverExt, Int256Executor};
use crate::memory_config::memory_config_with_fp;
use crate::pairing::{PairingExtension, PairingExtensionExecutor};
use openvm_circuit::{
    arch::{
        AirInventory, ChipInventoryError, InitFileGenerator, MatrixRecordArena, SystemConfig,
        VmBuilder, VmChipComplex, VmField, VmProverExtension,
    },
    system::{SystemChipInventory, SystemCpuBuilder, SystemExecutor},
};
use openvm_circuit_derive::{Executor, MeteredExecutor, VmConfig};
use openvm_cpu_backend::{CpuBackend, CpuDevice};
use openvm_sdk_config::TranspilerConfig;
use openvm_stark_backend::{StarkEngine, StarkProtocolConfig, Val, p3_field::PrimeField32};
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

pub mod algebra;
pub mod ecc;
pub mod int256;
pub mod pairing;

mod extension;
pub use extension::*;

pub mod memory_config;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_circuit::arch::DenseRecordArena;
        use openvm_circuit::system::cuda::{extensions::SystemGpuBuilder, SystemChipInventoryGPU};
        use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine, GpuBackend};
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
    // The optional extensions below are declared in the order their AIRs and chips are
    // registered. `CrushCpuBuilder` and the autoprecompile dummy chip complex must add them in
    // this same order, or the two inventories stop being index-aligned.
    /// Optional Int256 extension.
    #[extension(executor = "Int256Executor")]
    pub int256: Option<Int256>,
    /// Optional modular arithmetic extension, parameterised by the supported moduli.
    #[extension(executor = "ModularExtensionExecutor")]
    pub modular: Option<ModularExtension>,
    /// Optional Fp2 extension, parameterised by the supported moduli. Requires `modular`
    /// for the same moduli, as upstream does.
    #[extension(executor = "Fp2ExtensionExecutor")]
    pub fp2: Option<Fp2Extension>,
    /// Optional Weierstrass elliptic-curve extension, parameterised by the supported
    /// curves. Requires `modular` for each curve's coordinate field, as upstream does.
    #[extension(executor = "WeierstrassExtensionExecutor")]
    pub ecc: Option<WeierstrassExtension>,
    /// Optional pairing extension. This registers no AIRs -- it only supplies phantom
    /// hints -- so it needs no prover extension. The arithmetic a guest pairing
    /// implementation performs is proved by the modular, Fp2 and ecc chips, which must be
    /// enabled alongside it.
    #[extension(executor = "PairingExtensionExecutor<F>")]
    pub pairing: Option<PairingExtension>,
}

// This seems trivial but it's tricky to put into powdr-openvm because of some From implementation issues.
impl<
    F: PrimeField32 + openvm_stark_backend::p3_field::InjectiveMonomial<7>,
    ISA: OpenVmISA<Executor<F> = CrushConfigExecutor<F>>,
> From<CrushConfigExecutor<F>> for SpecializedExecutor<F, ISA>
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
            int256: None,
            modular: None,
            fp2: None,
            ecc: None,
            pairing: None,
        }
    }
}

impl CrushConfig {
    pub fn with_public_values(public_values: usize) -> Self {
        let system = system_config().with_public_values(public_values);
        Self {
            system,
            base: Default::default(),
            int256: None,
            modular: None,
            fp2: None,
            ecc: None,
            pairing: None,
        }
    }

    /// Whether any optional precompile extension is enabled, and therefore whether this
    /// config's AIR set differs from the default one. Callers that cache a proving key for
    /// the default config must generate a fresh one when this is true.
    ///
    /// `pairing` is included even though it adds no AIRs, so that adding one later cannot
    /// silently invalidate a cached key.
    pub fn has_optional_extensions(&self) -> bool {
        self.int256.is_some()
            || self.modular.is_some()
            || self.fp2.is_some()
            || self.ecc.is_some()
            || self.pairing.is_some()
    }

    /// Enable the Int256 precompiles.
    pub fn with_int256(mut self) -> Self {
        self.int256 = Some(Int256::default());
        self
    }

    /// Enable modular arithmetic for the given moduli.
    pub fn with_modular(mut self, supported_moduli: Vec<num_bigint::BigUint>) -> Self {
        self.modular = Some(ModularExtension { supported_moduli });
        self
    }

    /// Enable Fp2 arithmetic for the given (name, modulus) pairs. The same moduli must
    /// also be enabled via `with_modular`.
    pub fn with_fp2(mut self, supported_moduli: Vec<(String, num_bigint::BigUint)>) -> Self {
        self.fp2 = Some(Fp2Extension { supported_moduli });
        self
    }

    /// Enable Weierstrass curve arithmetic for the given curves. Each curve's coordinate
    /// field must also be enabled via `with_modular`.
    pub fn with_ecc(mut self, supported_curves: Vec<crate::ecc::CurveConfig>) -> Self {
        self.ecc = Some(WeierstrassExtension { supported_curves });
        self
    }

    /// Enable pairing hints for the given curves. Adds no AIRs; the accompanying
    /// arithmetic extensions must be enabled separately.
    pub fn with_pairing(mut self, supported_curves: Vec<crate::pairing::PairingCurve>) -> Self {
        self.pairing = Some(PairingExtension::new(supported_curves));
        self
    }

    pub fn with_public_values_and_segment_len(public_values: usize, segment_len: usize) -> Self {
        let system = system_config()
            .with_public_values(public_values)
            .with_max_segment_len(segment_len);
        Self {
            system,
            base: Default::default(),
            int256: None,
            modular: None,
            fp2: None,
            ecc: None,
            pairing: None,
        }
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
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
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
        if let Some(int256) = &config.int256 {
            VmProverExtension::<E, _, _>::extend_prover(&Int256CpuProverExt, int256, inventory)?;
        }
        if let Some(modular) = &config.modular {
            VmProverExtension::<E, _, _>::extend_prover(&AlgebraCpuProverExt, modular, inventory)?;
        }
        if let Some(fp2) = &config.fp2 {
            VmProverExtension::<E, _, _>::extend_prover(&AlgebraCpuProverExt, fp2, inventory)?;
        }
        if let Some(ecc) = &config.ecc {
            VmProverExtension::<E, _, _>::extend_prover(&EccCpuProverExt, ecc, inventory)?;
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
impl VmBuilder<BabyBearPoseidon2GpuEngine> for CrushGpuBuilder {
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
        // The Int256 chips are CPU-only here: no GPU prover extension has been ported.
        assert!(
            config.int256.is_none()
                && config.modular.is_none()
                && config.fp2.is_none()
                && config.ecc.is_none(),
            "the Int256, modular, Fp2 and ecc extensions are not supported on the CUDA backend"
        );

        let mut chip_complex = VmBuilder::<BabyBearPoseidon2GpuEngine>::create_chip_complex(
            &SystemGpuBuilder,
            &config.system,
            circuit,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<BabyBearPoseidon2GpuEngine, _, _>::extend_prover(
            &CrushGpuProverExt,
            &config.base,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
