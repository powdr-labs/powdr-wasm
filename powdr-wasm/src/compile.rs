//! Compile-to-disk pipelines for crush and RISC-V.

use std::path::Path;

use autoprecompiles::CrushISA;
use openvm_sdk::StdIn;
use openvm_sdk::config::{AggregationSystemParams, AppConfig};
use openvm_stark_sdk::config::{MAX_APP_LOG_STACKED_HEIGHT, app_params_with_100_bits_security};
use powdr_autoprecompiles::{GenerateConfig, PgoConfig, PgoType, SelectConfig};
use powdr_openvm::{
    StagedPipeline, execution_profile_from_guest,
    isa::OpenVmISA,
    make_default_empirical_constraints,
    program::{CompiledProgram, OriginalCompiledProgram},
};

use crate::proving::{AGG_PK_FILE, APP_PK_FILE, COMPILED_PROGRAM_FILE, CrushSdk, RiscvSdk};

/// Run powdr's staged APC pipeline (generate → select → setup) with no on-disk
/// artifact cache, returning the compiled program. Shared by the crush and
/// RISC-V compile/prove paths.
///
/// `select.autoprecompiles == 0` selects no autoprecompiles (`PgoType::None`);
/// otherwise candidates are ranked by cell density (`PgoType::Cell`) and the
/// top `select.autoprecompiles` are kept.
pub(crate) fn compile_with_pipeline<ISA: OpenVmISA>(
    original_program: OriginalCompiledProgram<'static, ISA>,
    stdin: StdIn,
    generate: GenerateConfig,
    select: SelectConfig,
) -> CompiledProgram<ISA> {
    let pgo_type = if select.autoprecompiles > 0 {
        PgoType::Cell
    } else {
        PgoType::None
    };
    let generate = generate.with_select_defaults(pgo_type, select);
    // No on-disk cache, so the hashing `inputs` / `max_columns` are irrelevant.
    let pgo_config = PgoConfig::new(pgo_type, None, Vec::new());
    StagedPipeline::new(original_program, None).setup(
        &generate,
        &pgo_config,
        select,
        move |guest, _inputs| execution_profile_from_guest(guest, stdin),
        make_default_empirical_constraints,
    )
}

/// Compile a crush program: load WASM, PGO, APC generation, keygen.
/// Saves the compiled program and proving keys to `output_dir`.
pub fn compile_crush_to_disk(
    original_program: OriginalCompiledProgram<'static, CrushISA>,
    stdin: StdIn,
    generate: GenerateConfig,
    select: SelectConfig,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;

    let apc_start = std::time::Instant::now();
    let compiled = compile_with_pipeline(original_program, stdin, generate, select);
    tracing::info!("APC generation took {:?}", apc_start.elapsed());

    // Serialize compiled program
    tracing::info!("Serializing compiled program...");
    let compiled_bytes = rmp_serde::to_vec(&compiled)?;
    std::fs::write(output_dir.join(COMPILED_PROGRAM_FILE), &compiled_bytes)?;
    tracing::info!(
        "Wrote compiled_program ({:.1} MB)",
        compiled_bytes.len() as f64 / 1e6
    );

    // Keygen
    let system_params = app_params_with_100_bits_security(MAX_APP_LOG_STACKED_HEIGHT);
    let app_config = AppConfig::new(compiled.vm_config.clone(), system_params);
    let sdk = CrushSdk::new_without_transpiler(app_config, AggregationSystemParams::default())?;

    let keygen_start = std::time::Instant::now();

    tracing::info!("Generating app proving key...");
    let app_pk = sdk.app_pk();
    let app_pk_bytes = rmp_serde::to_vec(app_pk)?;
    std::fs::write(output_dir.join(APP_PK_FILE), &app_pk_bytes)?;
    tracing::info!("Wrote app_pk ({:.1} MB)", app_pk_bytes.len() as f64 / 1e6);

    tracing::info!("Generating aggregation proving key...");
    let agg_pk = sdk.agg_pk();
    let agg_pk_bytes = rmp_serde::to_vec(&agg_pk)?;
    std::fs::write(output_dir.join(AGG_PK_FILE), &agg_pk_bytes)?;
    tracing::info!("Wrote agg_pk ({:.1} MB)", agg_pk_bytes.len() as f64 / 1e6);

    tracing::info!("Keygen took {:?}", keygen_start.elapsed());

    Ok(())
}

/// Compile a RISC-V program: build, PGO, APC generation, keygen.
/// Saves the compiled program and proving keys to `output_dir`.
pub fn compile_riscv_to_disk(
    program: &str,
    stdin: StdIn,
    generate: GenerateConfig,
    select: SelectConfig,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;

    let program_abs = std::fs::canonicalize(program).expect("Failed to resolve program path");
    let program_str = program_abs.to_str().unwrap();

    let original = powdr_openvm_riscv::compile_openvm(
        program_str,
        powdr_openvm_riscv::GuestOptions::default(),
    )
    .map_err(|e| eyre::eyre!("{e}"))?;

    let apc_start = std::time::Instant::now();
    let compiled = compile_with_pipeline(original, stdin, generate, select);
    tracing::info!("APC generation took {:?}", apc_start.elapsed());

    // Serialize compiled program
    tracing::info!("Serializing compiled RISC-V program...");
    let compiled_bytes = rmp_serde::to_vec(&compiled)?;
    std::fs::write(output_dir.join(COMPILED_PROGRAM_FILE), &compiled_bytes)?;
    tracing::info!(
        "Wrote compiled_program ({:.1} MB)",
        compiled_bytes.len() as f64 / 1e6
    );

    // Keygen
    let system_params = app_params_with_100_bits_security(MAX_APP_LOG_STACKED_HEIGHT);
    let app_config = AppConfig::new(compiled.vm_config.clone(), system_params);
    let sdk = RiscvSdk::new_without_transpiler(app_config, AggregationSystemParams::default())?;

    let keygen_start = std::time::Instant::now();

    tracing::info!("Generating app proving key...");
    let app_pk = sdk.app_pk();
    let app_pk_bytes = rmp_serde::to_vec(app_pk)?;
    std::fs::write(output_dir.join(APP_PK_FILE), &app_pk_bytes)?;
    tracing::info!("Wrote app_pk ({:.1} MB)", app_pk_bytes.len() as f64 / 1e6);

    tracing::info!("Generating aggregation proving key...");
    let agg_pk = sdk.agg_pk();
    let agg_pk_bytes = rmp_serde::to_vec(&agg_pk)?;
    std::fs::write(output_dir.join(AGG_PK_FILE), &agg_pk_bytes)?;
    tracing::info!("Wrote agg_pk ({:.1} MB)", agg_pk_bytes.len() as f64 / 1e6);

    tracing::info!("Keygen took {:?}", keygen_start.elapsed());

    Ok(())
}
