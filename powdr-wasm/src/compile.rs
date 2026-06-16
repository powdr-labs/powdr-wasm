//! Compile-to-disk pipelines for crush and RISC-V.

use std::path::Path;

use autoprecompiles::CrushISA;
use openvm_sdk::StdIn;
use openvm_sdk::config::{AggregationSystemParams, AppConfig};
use openvm_stark_sdk::config::{MAX_APP_LOG_STACKED_HEIGHT, app_params_with_100_bits_security};
use powdr_autoprecompiles::{
    GenerateConfig, PgoData, SelectConfig, empirical_constraints::EmpiricalConstraints,
};
use powdr_openvm::{
    customize_exe::{generate_apcs, select_apcs, setup},
    execution_profile_from_guest,
    program::OriginalCompiledProgram,
};

use crate::proving::{AGG_PK_FILE, APP_PK_FILE, COMPILED_PROGRAM_FILE, CrushSdk, RiscvSdk};

/// Compile a crush program: load WASM, PGO, APC generation, keygen.
/// Saves the compiled program and proving keys to `output_dir`.
pub fn compile_crush_to_disk(
    original_program: OriginalCompiledProgram<CrushISA>,
    stdin: StdIn,
    generate: GenerateConfig,
    select: SelectConfig,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(output_dir)?;

    let apc_start = std::time::Instant::now();
    let apc_count = select.autoprecompiles;

    let pgo_data = if apc_count > 0 {
        let execution_profile = execution_profile_from_guest(&original_program, stdin);
        PgoData::Cell(execution_profile, None)
    } else {
        PgoData::None
    };
    let generate = generate.with_select_defaults(pgo_data.pgo_type(), select);
    let degree_bound = generate.degree_bound;
    let ranked = generate_apcs(
        &original_program,
        &generate,
        pgo_data,
        EmpiricalConstraints::default(),
    );
    let apcs = select_apcs(ranked, select);
    let compiled = setup(original_program, apcs, degree_bound);
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
    let apc_count = select.autoprecompiles;
    let pgo_data = if apc_count > 0 {
        let execution_profile =
            powdr_openvm::execution_profile_from_guest(&original, stdin.clone());
        powdr_openvm_riscv::PgoData::Cell(execution_profile, None)
    } else {
        powdr_openvm_riscv::PgoData::None
    };
    let generate = generate.with_select_defaults(pgo_data.pgo_type(), select);
    let degree_bound = generate.degree_bound;
    let ranked = powdr_openvm_riscv::generate_apcs(
        &original,
        &generate,
        pgo_data,
        EmpiricalConstraints::default(),
    );
    let apcs = powdr_openvm_riscv::select_apcs(ranked, select);
    let compiled = powdr_openvm_riscv::setup(original, apcs, degree_bound);
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
