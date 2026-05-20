//! Proving infrastructure: engine setup, cached proving key, mock proof, and real proof.

use std::path::Path;
use std::sync::OnceLock;

use autoprecompiles::CrushISA;
use crush_circuit::{CrushConfig, CrushCpuBuilder};
use openvm_circuit::arch::{
    Executor, MeteredExecutor, PreflightExecutor, VirtualMachine, VmBuilder, VmCircuitConfig,
    VmExecutionConfig, VmState, debug_proving_ctx,
};
use openvm_instructions::exe::VmExe;
use openvm_sdk::StdIn;
use openvm_sdk::config::{AppConfig, DEFAULT_APP_LOG_BLOWUP};
use openvm_sdk::keygen::{AggProvingKey, AppProvingKey};
use openvm_sdk::prover::verify_app_proof;
use openvm_stark_backend::{config::Val, p3_field::PrimeField32};
use openvm_stark_sdk::{
    config::{
        FriParameters,
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
    },
    engine::{StarkEngine, StarkFriEngine},
    openvm_stark_backend::{
        keygen::types::MultiStarkProvingKey, prover::hal::DeviceDataTransporter,
    },
};
use powdr_autoprecompiles::{
    PowdrConfig,
    empirical_constraints::EmpiricalConstraints,
    pgo::{CellPgo, NonePgo},
};
use powdr_openvm::extraction_utils::OriginalVmConfig;
use powdr_openvm::program::CompiledProgram;
use powdr_openvm::{DEFAULT_DEGREE_BOUND, SpecializedConfig};
use powdr_openvm::{
    customize_exe::{OpenVmApcCandidate, customize},
    execution_profile_from_guest,
    program::OriginalCompiledProgram,
};

pub type F = openvm_stark_sdk::p3_baby_bear::BabyBear;
type SC = BabyBearPoseidon2Config;

static VM_PROVING_KEY: OnceLock<MultiStarkProvingKey<SC>> = OnceLock::new();

/// Create a CPU engine with default FRI parameters.
pub fn cpu_engine() -> BabyBearPoseidon2Engine {
    let fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    BabyBearPoseidon2Engine::new(fri_params)
}

// --- Test-only backend infrastructure ---
// These are only used by integration tests (binary crate, so `pub` alone doesn't suppress warnings).

/// Proving backend selection.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    #[cfg(feature = "cuda")]
    Gpu,
}

#[cfg(test)]
impl Backend {
    /// Run mock proof on this backend, returning the final state.
    pub fn mock_prove(
        self,
        exe: &VmExe<F>,
        init_state: VmState<F>,
    ) -> Result<VmState<F>, Box<dyn std::error::Error>> {
        match self {
            Backend::Cpu => mock_prove(exe, init_state),
            #[cfg(feature = "cuda")]
            Backend::Gpu => mock_prove_gpu(exe, init_state),
        }
    }

    /// Name for display/logging.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            #[cfg(feature = "cuda")]
            Backend::Gpu => "GPU",
        }
    }
}

/// All available backends for the current build configuration.
#[cfg(test)]
pub const ALL_BACKENDS: &[Backend] = &[
    Backend::Cpu,
    #[cfg(feature = "cuda")]
    Backend::Gpu,
];

/// Alias for backwards compatibility.
#[cfg(test)]
pub fn default_engine() -> BabyBearPoseidon2Engine {
    cpu_engine()
}

/// Create a GPU engine with default FRI parameters.
#[cfg(feature = "cuda")]
pub fn gpu_engine() -> openvm_cuda_backend::engine::GpuBabyBearPoseidon2Engine {
    let fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    openvm_cuda_backend::engine::GpuBabyBearPoseidon2Engine::new(fri_params)
}

pub fn vm_proving_key() -> &'static MultiStarkProvingKey<SC> {
    VM_PROVING_KEY.get_or_init(|| {
        let config = CrushConfig::default();
        let engine = cpu_engine();
        let circuit = config
            .create_airs()
            .expect("failed to create AIR inventory for keygen");
        circuit.keygen(&engine)
    })
}

#[cfg(not(feature = "cuda"))]
pub(crate) type CrushSdk = powdr_openvm::PowdrSdkCpu<CrushISA>;

#[cfg(feature = "cuda")]
pub(crate) type CrushSdk = powdr_openvm::PowdrSdkGpu<CrushISA>;

#[cfg(not(feature = "cuda"))]
pub(crate) type RiscvSdk = powdr_openvm::PowdrSdkCpu<powdr_openvm_riscv::RiscvISA>;

#[cfg(feature = "cuda")]
pub(crate) type RiscvSdk = powdr_openvm::PowdrSdkGpu<powdr_openvm_riscv::RiscvISA>;

pub(crate) const APP_PK_FILE: &str = "app_pk.bin";
pub(crate) const AGG_PK_FILE: &str = "agg_pk.bin";
pub(crate) const COMPILED_PROGRAM_FILE: &str = "compiled_program.bin";

fn default_app_config_without_apcs() -> AppConfig<SpecializedConfig<CrushISA>> {
    let vm_config = CrushConfig::default();
    let app_config = powdr_openvm::SpecializedConfig::<CrushISA>::new(
        OriginalVmConfig::new(vm_config),
        vec![],
        DEFAULT_DEGREE_BOUND,
    );

    let app_fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    AppConfig::new(app_fri_params, app_config)
}

/// Generate app and aggregation proving keys and write them to `cache_dir`.
pub fn keygen_to_disk(cache_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(cache_dir)?;

    let app_config = default_app_config_without_apcs();
    let sdk = CrushSdk::new_without_transpiler(app_config)?;

    tracing::info!("Generating app proving key...");
    let app_pk = sdk.app_pk();
    let app_pk_bytes = rmp_serde::to_vec(app_pk)?;
    std::fs::write(cache_dir.join(APP_PK_FILE), &app_pk_bytes)?;
    tracing::info!("Wrote app_pk ({:.1} MB)", app_pk_bytes.len() as f64 / 1e6);

    tracing::info!("Generating aggregation proving key...");
    let agg_pk = sdk.agg_pk();
    let agg_pk_bytes = rmp_serde::to_vec(agg_pk)?;
    std::fs::write(cache_dir.join(AGG_PK_FILE), &agg_pk_bytes)?;
    tracing::info!("Wrote agg_pk ({:.1} MB)", agg_pk_bytes.len() as f64 / 1e6);

    Ok(())
}

fn build_sdk(
    cache_dir: Option<&Path>,
    max_segment_len: Option<u32>,
) -> Result<CrushSdk, Box<dyn std::error::Error>> {
    let mut app_config = default_app_config_without_apcs();
    if let Some(len) = max_segment_len {
        app_config
            .app_vm_config
            .as_mut()
            .segmentation_limits
            .set_max_trace_height(len);
    }
    let sdk = CrushSdk::new_without_transpiler(app_config)?;

    if let Some(dir) = cache_dir {
        let app_pk_path = dir.join(APP_PK_FILE);
        tracing::info!("Loading cached app_pk from {}", app_pk_path.display());
        let app_pk: AppProvingKey<SpecializedConfig<CrushISA>> =
            rmp_serde::from_slice(&std::fs::read(&app_pk_path)?)?;
        sdk.set_app_pk(app_pk).map_err(|_| "app_pk already set")?;

        let agg_pk_path = dir.join(AGG_PK_FILE);
        if agg_pk_path.exists() {
            tracing::info!("Loading cached agg_pk from {}", agg_pk_path.display());
            let agg_pk: AggProvingKey = rmp_serde::from_slice(&std::fs::read(&agg_pk_path)?)?;
            sdk.set_agg_pk(agg_pk).map_err(|_| "agg_pk already set")?;
        }
    }

    Ok(sdk)
}

/// Generate and verify a real cryptographic proof, with optional recursion.
pub fn prove(
    original_program: OriginalCompiledProgram<CrushISA>,
    stdin: StdIn,
    recursion: bool,
    powdr_config: PowdrConfig,
    cache_dir: Option<&Path>,
    max_segment_len: Option<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let apc_count = powdr_config.autoprecompiles;
    let compiled = if apc_count > 0 {
        let execution_profile = execution_profile_from_guest(&original_program, stdin.clone());
        customize(
            original_program,
            powdr_config,
            CellPgo::<_, OpenVmApcCandidate<CrushISA>>::with_pgo_data_and_max_columns(
                execution_profile,
                None,
            ),
            EmpiricalConstraints::default(),
        )
    } else {
        customize(
            original_program,
            powdr_config,
            NonePgo::default(),
            EmpiricalConstraints::default(),
        )
    };
    let app_fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    let mut app_config = AppConfig::new(app_fri_params, compiled.vm_config.clone());
    if let Some(len) = max_segment_len {
        app_config
            .app_vm_config
            .as_mut()
            .segmentation_limits
            .set_max_trace_height(len);
    }
    let sdk = if apc_count == 0 {
        build_sdk(cache_dir, max_segment_len)?
    } else {
        CrushSdk::new_without_transpiler(app_config)?
    };

    let mut app_prover = sdk.app_prover(compiled.exe.clone())?;
    let app_proof = app_prover.prove(stdin)?;

    let app_vk = sdk.app_pk().get_app_vk();
    verify_app_proof(&app_vk, &app_proof)?;

    if recursion {
        let mut agg_prover = sdk.prover(compiled.exe.clone())?.agg_prover;

        // Note that this proof is not verified. We assume that any valid app proof
        // (verified above) also leads to a valid aggregation proof.
        // If this was not the case, it would be a completeness bug in OpenVM.
        let start = std::time::Instant::now();
        let _ = agg_prover.generate_root_verifier_input(app_proof)?;
        tracing::info!("Agg proof (inner recursion) took {:?}", start.elapsed());
    }

    Ok(())
}

/// Mock proof with constraint verification (all segments) using a specific engine and builder.
/// Returns the final state after all segments have been processed.
pub fn mock_prove_with<E, VB>(
    engine: E,
    builder: VB,
    exe: &VmExe<F>,
    init_state: VmState<F>,
) -> Result<VmState<F>, Box<dyn std::error::Error>>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E, VmConfig = CrushConfig> + Clone,
    Val<E::SC>: PrimeField32,
    <CrushConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
        + MeteredExecutor<Val<E::SC>>
        + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
{
    let pk = vm_proving_key();
    let d_pk = engine.device().transport_pk_to_device(pk);
    let vm_config = CrushConfig::default();
    let mut vm = VirtualMachine::<_, VB>::new(engine, builder, vm_config, d_pk)?;

    // Run metered execution to discover segments.
    let metered_ctx = vm.build_metered_ctx(exe);
    let metered_instance = vm.metered_interpreter(exe)?;
    let (segments, _) =
        metered_instance.execute_metered_from_state(init_state.clone(), metered_ctx)?;

    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    vm.load_program(cached_program_trace);

    // Preflight + constraint verification per segment.
    let mut preflight_interpreter = vm.preflight_interpreter(exe)?;
    let mut state = init_state;
    for segment in &segments {
        vm.transport_init_memory_to_device(&state.memory);
        let preflight_output = vm.execute_preflight(
            &mut preflight_interpreter,
            state,
            Some(segment.num_insns),
            &segment.trace_heights,
        )?;
        state = preflight_output.to_state;

        let ctx = vm.generate_proving_ctx(
            preflight_output.system_records,
            preflight_output.record_arenas,
        )?;
        debug_proving_ctx(&vm, pk, &ctx);
    }

    Ok(state)
}

/// Mock proof with constraint verification (all segments) using CPU engine.
/// Returns the final state after all segments have been processed.
pub fn mock_prove(
    exe: &VmExe<F>,
    init_state: VmState<F>,
) -> Result<VmState<F>, Box<dyn std::error::Error>> {
    mock_prove_with(cpu_engine(), CrushCpuBuilder, exe, init_state)
}

/// Mock proof with constraint verification (all segments) using GPU engine.
/// Uses the same CrushConfig as CPU but with CrushGpuBuilder for GPU tracegen.
/// Returns the final state after all segments have been processed.
#[cfg(feature = "cuda")]
pub fn mock_prove_gpu(
    exe: &VmExe<F>,
    init_state: VmState<F>,
) -> Result<VmState<F>, Box<dyn std::error::Error>> {
    mock_prove_with(
        gpu_engine(),
        crush_circuit::CrushGpuBuilder,
        exe,
        init_state,
    )
}

/// Prove from a pre-compiled crush artifact directory.
pub fn prove_from_compiled(
    compiled_dir: &Path,
    stdin: StdIn,
    recursion: bool,
    max_segment_len: Option<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("Loading compiled program...");
    let compiled: CompiledProgram<CrushISA> =
        rmp_serde::from_slice(&std::fs::read(compiled_dir.join(COMPILED_PROGRAM_FILE))?)?;

    let app_fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    let mut app_config = AppConfig::new(app_fri_params, compiled.vm_config.clone());
    if let Some(len) = max_segment_len {
        app_config
            .app_vm_config
            .as_mut()
            .segmentation_limits
            .set_max_trace_height(len);
    }
    let sdk = CrushSdk::new_without_transpiler(app_config)?;

    tracing::info!("Loading cached app_pk...");
    let app_pk: AppProvingKey<SpecializedConfig<CrushISA>> =
        rmp_serde::from_slice(&std::fs::read(compiled_dir.join(APP_PK_FILE))?)?;
    sdk.set_app_pk(app_pk).map_err(|_| "app_pk already set")?;

    if recursion {
        tracing::info!("Loading cached agg_pk...");
        let agg_pk: AggProvingKey =
            rmp_serde::from_slice(&std::fs::read(compiled_dir.join(AGG_PK_FILE))?)?;
        sdk.set_agg_pk(agg_pk).map_err(|_| "agg_pk already set")?;
    }

    let mut app_prover = sdk.app_prover(compiled.exe.clone())?;
    let app_proof = app_prover.prove(stdin)?;

    let app_vk = sdk.app_pk().get_app_vk();
    verify_app_proof(&app_vk, &app_proof)?;

    if recursion {
        let mut agg_prover = sdk.prover(compiled.exe.clone())?.agg_prover;
        let start = std::time::Instant::now();
        let _ = agg_prover.generate_root_verifier_input(app_proof)?;
        tracing::info!("Agg proof (inner recursion) took {:?}", start.elapsed());
    }

    Ok(())
}

/// Prove from a pre-compiled RISC-V artifact directory.
pub fn prove_riscv_from_compiled(
    compiled_dir: &Path,
    stdin: StdIn,
    recursion: bool,
    max_segment_len: Option<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("Loading compiled RISC-V program...");
    let compiled: CompiledProgram<powdr_openvm_riscv::RiscvISA> =
        rmp_serde::from_slice(&std::fs::read(compiled_dir.join(COMPILED_PROGRAM_FILE))?)?;

    let app_fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_APP_LOG_BLOWUP);
    let mut app_config = AppConfig::new(app_fri_params, compiled.vm_config.clone());
    if let Some(len) = max_segment_len {
        app_config
            .app_vm_config
            .as_mut()
            .segmentation_limits
            .set_max_trace_height(len);
    }
    let sdk = RiscvSdk::new_without_transpiler(app_config)?;

    tracing::info!("Loading cached app_pk...");
    let app_pk: AppProvingKey<SpecializedConfig<powdr_openvm_riscv::RiscvISA>> =
        rmp_serde::from_slice(&std::fs::read(compiled_dir.join(APP_PK_FILE))?)?;
    sdk.set_app_pk(app_pk).map_err(|_| "app_pk already set")?;

    let mut app_prover = sdk.app_prover(compiled.exe.clone())?;
    let app_proof = app_prover.prove(stdin)?;

    let app_vk = sdk.app_pk().get_app_vk();
    verify_app_proof(&app_vk, &app_proof)?;

    if recursion {
        tracing::info!("Loading cached agg_pk...");
        let agg_pk: AggProvingKey =
            rmp_serde::from_slice(&std::fs::read(compiled_dir.join(AGG_PK_FILE))?)?;
        sdk.set_agg_pk(agg_pk).map_err(|_| "agg_pk already set")?;

        let mut agg_prover = sdk.prover(compiled.exe.clone())?.agg_prover;
        let start = std::time::Instant::now();
        let _ = agg_prover.generate_root_verifier_input(app_proof)?;
        tracing::info!("Agg proof (inner recursion) took {:?}", start.elapsed());
    }

    Ok(())
}
