//! Test-only helpers: metered execution and preflight.
//! For mock proof, delegates to `crate::proving::mock_prove`.

use crush_circuit::{CrushConfig, CrushCpuBuilder};
use openvm_circuit::arch::{VirtualMachine, VmState, execution_mode::Segment};
use openvm_instructions::exe::VmExe;
use openvm_stark_sdk::openvm_stark_backend::{StarkEngine, prover::DeviceDataTransporter};

use std::borrow::Cow;

use openvm_stark_sdk::openvm_stark_backend::keygen::types::MultiStarkProvingKey;

use crate::proving::{F, SC, default_engine, keygen, vm_proving_key};

/// Proving key for `vm_config`: the cached default-config one where possible, a fresh one when
/// an optional extension changes the AIR set.
fn proving_key<E: StarkEngine<SC = SC>>(
    engine: &E,
    vm_config: &CrushConfig,
) -> Cow<'static, MultiStarkProvingKey<SC>> {
    if vm_config.has_optional_extensions() {
        Cow::Owned(keygen(engine, vm_config))
    } else {
        Cow::Borrowed(vm_proving_key())
    }
}

/// Metered execution. Returns (segments, final_state).
pub fn test_metered_execution(
    vm_config: CrushConfig,
    exe: &VmExe<F>,
    initial_state: VmState<F>,
) -> Result<(Vec<Segment>, VmState<F>), Box<dyn std::error::Error>> {
    let engine = default_engine();
    let pk = proving_key(&engine, &vm_config);
    let d_pk = engine.device().transport_pk_to_device(&pk);
    let vm = VirtualMachine::<_, CrushCpuBuilder>::new(engine, CrushCpuBuilder, vm_config, d_pk)?;

    let metered_ctx = vm.build_metered_ctx(exe);
    let metered_instance = vm.metered_interpreter(exe)?;
    let (segments, final_state) =
        metered_instance.execute_metered_from_state(initial_state, metered_ctx)?;

    Ok((segments, final_state))
}

/// Preflight (all segments). Returns the final state after the last segment.
pub fn test_preflight(
    vm_config: CrushConfig,
    exe: &VmExe<F>,
    initial_state: VmState<F>,
) -> Result<VmState<F>, Box<dyn std::error::Error>> {
    let engine = default_engine();
    let pk = proving_key(&engine, &vm_config);
    let d_pk = engine.device().transport_pk_to_device(&pk);
    let vm = VirtualMachine::<_, CrushCpuBuilder>::new(engine, CrushCpuBuilder, vm_config, d_pk)?;

    // Run metered execution to discover segments.
    let metered_ctx = vm.build_metered_ctx(exe);
    let metered_instance = vm.metered_interpreter(exe)?;
    let (segments, _) =
        metered_instance.execute_metered_from_state(initial_state.clone(), metered_ctx)?;

    // Preflight each segment.
    let mut preflight_interpreter = vm.preflight_interpreter(exe)?;
    let mut state = initial_state;
    for segment in &segments {
        let output = vm.execute_preflight(
            &mut preflight_interpreter,
            state,
            Some(segment.num_insns),
            &segment.trace_heights,
        )?;
        state = output.to_state;
    }

    Ok(state)
}
