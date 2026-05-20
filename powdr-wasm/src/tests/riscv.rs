use openvm_sdk::StdIn;
use powdr_openvm_riscv::{GuestOptions, PgoConfig};

/// Compile and execute an OpenVM RISC-V guest program via powdr-openvm.
fn run_openvm_guest(guest: &str, args: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
    crate::setup_tracing_with_log_level(tracing::Level::WARN);

    let guest_abs = std::fs::canonicalize(guest)?;
    let guest_str = guest_abs.to_str().unwrap();

    let original = powdr_openvm_riscv::compile_openvm(guest_str, GuestOptions::default())?;

    let config = powdr_openvm::default_powdr_openvm_config(0, 0);
    let compiled = powdr_openvm_riscv::compile_exe(
        original,
        config,
        PgoConfig::None,
        powdr_autoprecompiles::empirical_constraints::EmpiricalConstraints::default(),
    )?;

    let mut stdin = StdIn::default();
    for arg in args {
        stdin.write(arg);
    }

    // TODO the outputs are not checked yet because powdr-openvm does not return the outputs.
    powdr_openvm::execute(compiled, stdin)?;

    Ok(())
}

#[test]
fn test_keccak_rust_openvm() {
    let path = format!(
        "{}/../sample-programs/keccak_with_inputs",
        env!("CARGO_MANIFEST_DIR")
    );
    // TODO When the function above does check the outputs, pass expectation &[41].
    run_openvm_guest(&path, &[1]).unwrap();
}

#[test]
fn test_keccak_precompile_openvm() {
    let path = format!(
        "{}/../sample-programs/keccak_precompile",
        env!("CARGO_MANIFEST_DIR")
    );
    run_openvm_guest(&path, &[1]).unwrap();
}

#[test]
fn test_u256_openvm() {
    let path = format!(
        "{}/../sample-programs/u256_matmul",
        env!("CARGO_MANIFEST_DIR")
    );
    run_openvm_guest(&path, &[1]).unwrap();
}
