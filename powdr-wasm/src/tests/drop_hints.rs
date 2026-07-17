//! Tests for the drop (liveness) hints feature: hints emitted by the guest
//! translation let APC generation elide fully-internal register accesses.
//!
//! Both runs share the same guest and PGO profile; only
//! [`GenerateConfig::with_drop_hints`] differs, mirroring the CLI's
//! `--disable-drop-hints` flag.

use std::collections::BTreeMap;

use openvm_sdk::StdIn;
use powdr_autoprecompiles::{PgoConfig, PgoType};
use powdr_openvm::{
    RankedApcs, StagedPipeline, execution_profile_from_guest, make_default_empirical_constraints,
};

use autoprecompiles::CrushISA;

/// Per-block observables of a generated APC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ApcStats {
    bus_interactions: usize,
    memory_drops: usize,
}

/// Generate all APC candidates for the keccak sample guest, with or without
/// drop hints, and return their stats keyed by block start PC.
fn generate_apc_stats(use_drop_hints: bool) -> BTreeMap<u64, ApcStats> {
    let original =
        crate::load_wasm_original_program("../sample-programs/keccak.wasm", "main", false);

    let mut stdin: StdIn = StdIn::default();
    stdin.write(&0u32);
    stdin.write(&0u32);
    let inputs = rmp_serde::to_vec(&stdin).expect("failed to serialize profiling input");

    let generate = powdr_openvm::default_generate_config()
        .with_superblocks(1, None, None)
        .with_drop_hints(use_drop_hints);
    let pgo_config = PgoConfig::new(PgoType::Cell, None, inputs);

    let pipeline = StagedPipeline::new(original, None);
    let ranked: RankedApcs<CrushISA> = pipeline.generate_apcs(
        &generate,
        &pgo_config,
        |guest, inputs| {
            let stdin: StdIn =
                rmp_serde::from_slice(inputs).expect("failed to deserialize profiling input");
            execution_profile_from_guest(guest, stdin)
        },
        make_default_empirical_constraints,
    );

    ranked
        .iter()
        .map(|apc_with_stats| {
            let apc = apc_with_stats.apc();
            let machine = apc.machine();
            (
                apc.start_pcs()[0],
                ApcStats {
                    bus_interactions: machine.bus_interactions.len(),
                    memory_drops: machine.memory_drops.len(),
                },
            )
        })
        .collect()
}

#[test]
fn drop_hints_reduce_apc_bus_interactions() {
    crate::setup_tracing_with_log_level(tracing::Level::WARN);

    let with_hints = generate_apc_stats(true);
    let without_hints = generate_apc_stats(false);

    // Block detection is hint-independent: both runs must see the same blocks.
    assert_eq!(
        with_hints.keys().collect::<Vec<_>>(),
        without_hints.keys().collect::<Vec<_>>(),
        "hint usage must not change the set of APC candidate blocks"
    );

    // Disabled: no hints may reach any machine.
    let stray = without_hints
        .values()
        .map(|s| s.memory_drops)
        .sum::<usize>();
    assert_eq!(
        stray, 0,
        "disable_drop_hints left {stray} memory drops in machines"
    );

    // Enabled: hints must flow through to the machines.
    let hints = with_hints.values().map(|s| s.memory_drops).sum::<usize>();
    assert!(
        hints > 0,
        "no memory drops reached any machine with hints enabled"
    );

    // The feature must strictly reduce the total optimized bus interactions.
    let bi_with = with_hints
        .values()
        .map(|s| s.bus_interactions)
        .sum::<usize>();
    let bi_without = without_hints
        .values()
        .map(|s| s.bus_interactions)
        .sum::<usize>();
    assert!(
        bi_with < bi_without,
        "drop hints did not reduce bus interactions: {bi_with} (with) vs {bi_without} (without)"
    );
}
