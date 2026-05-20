//! Isolated testing infrastructure for crush instructions.
//!
//! This module provides a framework for testing crush instructions
//! through isolated execution:
//! - Raw execution (InterpretedInstance::execute_from_state)
//! - Metered execution (InterpretedInstance::execute_metered_from_state)
//! - Preflight (VirtualMachine::execute_preflight)
//! - Proof generation (VirtualMachine::prove)

use crush_circuit::{CrushConfig, adapters::RV32_REGISTER_NUM_LIMBS, memory_config::FpMemory};
use openvm_circuit::{
    arch::{ExecutionError, VmExecutor, VmState},
    system::memory::online::GuestMemory,
};
use openvm_instructions::{
    exe::VmExe,
    instruction::Instruction,
    program::{DEFAULT_PC_STEP, Program},
    riscv::RV32_REGISTER_AS,
};
use openvm_sdk::StdIn;

use super::helpers;
use crate::proving::{ALL_BACKENDS, Backend};
use crush_translation::instruction_builder::*;

type F = openvm_stark_sdk::p3_baby_bear::BabyBear;

/// Memory address space for RAM (heap memory).
const RV32_MEMORY_AS: u32 = openvm_instructions::riscv::RV32_MEMORY_AS;

/// Specification for an test case.
/// Defines the start state, program, and expected end state.
#[derive(Clone, Default)]
pub struct TestSpec {
    /// The program to execute (should end with halt instruction).
    pub program: Vec<Instruction<F>>,

    /// Initial PC (default: 0).
    pub start_pc: u32,
    /// Initial FP (default: 0).
    pub start_fp: u32,
    /// Initial register values: (register_index, value).
    pub start_registers: Vec<(usize, u32)>,
    /// Initial RAM values: (address, value).
    pub start_ram: Vec<(u32, u32)>,

    /// Expected PC after execution (if None, start_pc + num_instructions * DEFAULT_PC_STEP).
    pub expected_pc: Option<u32>,
    /// Expected FP after execution (if None, start_fp).
    pub expected_fp: Option<u32>,
    /// Expected register values after execution: (register_index, value).
    pub expected_registers: Vec<(usize, u32)>,
    /// Expected RAM values after execution: (address, value).
    pub expected_ram: Vec<(u32, u32)>,

    /// Register indices whose values are raw FP pointers (e.g., saved FP in call/ret tests).
    /// When rebasing, these values are shifted by `delta * RV32_REGISTER_NUM_LIMBS`.
    /// Indices are in the original (pre-rebase) namespace.
    pub fp_value_registers: Vec<usize>,

    /// Optional stdin data for hint tests.
    pub stdin: StdIn,
}

/// Read a register value from memory.
fn read_register(memory: &GuestMemory, reg: usize) -> u32 {
    // The instruction builder multiplies all registers by RV32_REGISTER_NUM_LIMBS, so this is the
    // address that actually hits the key/value store.
    let addr = (reg * RV32_REGISTER_NUM_LIMBS) as u32;
    let bytes: [u8; 4] = unsafe { memory.read(RV32_REGISTER_AS, addr) };
    u32::from_le_bytes(bytes)
}

/// Read a 32-bit value from RAM.
fn read_ram(memory: &GuestMemory, addr: u32) -> u32 {
    let bytes: [u8; 4] = unsafe { memory.read(RV32_MEMORY_AS, addr) };
    u32::from_le_bytes(bytes)
}

/// Read FP from memory.
fn read_fp(memory: &GuestMemory) -> u32 {
    memory.fp::<F>()
}

/// Build a VmExe from a test specification (program and PC only).
fn build_exe(spec: &TestSpec) -> VmExe<F> {
    let program = Program::from_instructions(&spec.program);
    VmExe::new(program).with_pc_start(spec.start_pc)
}

/// Create initial VmState from spec, exe, and config.
/// Sets up memory with initial register values, RAM values, and FP from the spec.
fn build_initial_state(spec: &TestSpec, exe: &VmExe<F>, vm_config: &CrushConfig) -> VmState<F> {
    let mut state = VmState::initial(
        &vm_config.system,
        &exe.init_memory,
        exe.pc_start,
        spec.stdin.clone(),
    );
    state.metrics.debug_infos = exe.program.debug_infos();

    // Set initial registers
    for &(reg, value) in &spec.start_registers {
        // The instruction builder multiplies all registers by RV32_REGISTER_NUM_LIMBS, so this is the
        // address that actually hits the key/value store.
        let addr = (reg * RV32_REGISTER_NUM_LIMBS) as u32;
        unsafe {
            state
                .memory
                .write::<u8, 4>(RV32_REGISTER_AS, addr, value.to_le_bytes());
        }
    }

    // Set initial RAM
    for &(addr, value) in &spec.start_ram {
        unsafe {
            state
                .memory
                .write::<u8, 4>(RV32_MEMORY_AS, addr, value.to_le_bytes());
        }
    }

    // Set initial FP
    // Again, the raw value stored in the state is the FP multiplied by RV32_REGISTER_NUM_LIMBS.
    state
        .memory
        .set_fp::<F>(spec.start_fp * RV32_REGISTER_NUM_LIMBS as u32);

    state
}

/// Verify the final state matches expected values.
fn verify_state(
    spec: &TestSpec,
    final_state: &VmState<F>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Verify expected registers
    for &(reg, expected_value) in &spec.expected_registers {
        let actual_value = read_register(&final_state.memory, reg);
        if actual_value != expected_value {
            return Err(format!("reg[{reg}] expected {expected_value}, got {actual_value}").into());
        }
    }

    // Verify expected RAM
    for &(addr, expected_value) in &spec.expected_ram {
        let actual_value = read_ram(&final_state.memory, addr);
        if actual_value != expected_value {
            return Err(
                format!("RAM[{addr}] expected {expected_value}, got {actual_value}").into(),
            );
        }
    }

    // Verify expected FP
    // The raw value stored in the state is the FP multiplied by RV32_REGISTER_NUM_LIMBS, so we need to divide it to get the actual FP.
    let actual_fp = read_fp(&final_state.memory) / RV32_REGISTER_NUM_LIMBS as u32;
    let expected_fp = spec.expected_fp.unwrap_or(spec.start_fp);
    if actual_fp != expected_fp {
        return Err(format!("FP expected {expected_fp}, got {actual_fp}").into());
    }

    // Verify expected PC
    let actual_pc = final_state.pc();
    let expected_pc = spec
        .expected_pc
        // The added HALT instruction does not advance the PC!
        .unwrap_or(spec.start_pc + ((spec.program.len() - 1) as u32 * DEFAULT_PC_STEP));
    if actual_pc != expected_pc {
        return Err(format!("PC expected {expected_pc}, got {actual_pc}").into());
    }

    Ok(())
}

/// Raw execution using InterpretedInstance::execute_from_state.
pub fn test_execution(spec: &TestSpec) -> Result<(), Box<dyn std::error::Error>> {
    let exe = build_exe(spec);
    let vm_config = CrushConfig::default();
    let vm = VmExecutor::new(vm_config.clone()).unwrap();
    let instance = vm.instance(&exe).unwrap();

    let from_state = build_initial_state(spec, &exe, &vm_config);
    let final_state = instance.execute_from_state(from_state, None)?;

    verify_state(spec, &final_state)
}

/// Metered execution using InterpretedInstance::execute_metered_from_state.
pub fn test_metered_execution(spec: &TestSpec) -> Result<(), Box<dyn std::error::Error>> {
    let exe = build_exe(spec);
    let vm_config = CrushConfig::default();
    let (segments, final_state) = helpers::test_metered_execution(
        vm_config.clone(),
        &exe,
        build_initial_state(spec, &exe, &vm_config),
    )?;

    assert_eq!(segments.len(), 1, "expected a single segment");
    verify_state(spec, &final_state)
}

/// Preflight using VirtualMachine::execute_preflight.
pub fn test_preflight(spec: &TestSpec) -> Result<(), Box<dyn std::error::Error>> {
    let exe = build_exe(spec);
    let vm_config = CrushConfig::default();
    let final_state = helpers::test_preflight(
        vm_config.clone(),
        &exe,
        build_initial_state(spec, &exe, &vm_config),
    )?;

    verify_state(spec, &final_state)
}

/// Mock proving with constraint verification using debug_proving_ctx.
/// This generates traces and verifies all constraints are satisfied without
/// generating actual cryptographic proofs.
/// Runs on the specified backends (CPU, GPU, or both) and verifies the final state.
pub fn test_prove(spec: &TestSpec, backends: &[Backend]) -> Result<(), Box<dyn std::error::Error>> {
    let exe = build_exe(spec);
    let vm_config = CrushConfig::default();
    let init_state = build_initial_state(spec, &exe, &vm_config);

    for &backend in backends {
        let final_state = backend
            .mock_prove(CrushConfig::default(), &exe, init_state.clone())
            .map_err(|e| format!("{} mock_prove: {e}", backend.name()))?;
        verify_state(spec, &final_state)
            .map_err(|e| format!("{} verify_state: {e}", backend.name()))?;
    }
    Ok(())
}

/// Run a test spec through all stages (execution, metered, preflight, prove)
/// on all available backends.
fn test_spec(spec: TestSpec) {
    test_spec_with_backends(spec, ALL_BACKENDS)
}

/// Run a test spec through all stages with specified proving backends.
/// Appends halt instruction automatically.
pub fn test_spec_with_backends(mut spec: TestSpec, backends: &[Backend]) {
    // Append halt instruction:
    spec.program.push(halt());

    // Test all stages and collect errors
    let mut is_error = false;
    if let Err(e) = test_execution(&spec) {
        println!("test_execution: {e}");
        is_error = true;
    }
    if let Err(e) = test_metered_execution(&spec) {
        println!("test_metered_execution: {e}");
        is_error = true;
    }
    if let Err(e) = test_preflight(&spec) {
        println!("test_preflight: {e}");
        is_error = true;
    }
    if let Err(e) = test_prove(&spec, backends) {
        println!("test_prove: {e}");
        is_error = true;
    }

    assert!(!is_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup_tracing_with_log_level;
    use tracing::Level;

    const REG_BASE_FITS_1_BYTE: u32 = 200;
    const REG_BASE_FITS_2_BYTES: u32 = 30_000;
    const REG_BASE_FITS_3_BYTES: u32 = 3_000_000;

    fn rebase_spec(spec: &TestSpec, new_base: u32) -> TestSpec {
        let mut rebased = spec.clone();
        let old_base = spec.start_fp;
        let delta = new_base as i64 - old_base as i64;
        let raw_delta = delta * RV32_REGISTER_NUM_LIMBS as i64;

        let shift_register = |register_index: usize| -> usize {
            let shifted = register_index as i64 + delta;
            assert!(
                shifted >= 0,
                "register index underflow after rebasing: {register_index} -> {shifted}"
            );
            shifted as usize
        };

        let shift_reg_entry = |register_index: usize, value: u32| -> (usize, u32) {
            let new_value = if spec.fp_value_registers.contains(&register_index) {
                (value as i64 + raw_delta) as u32
            } else {
                value
            };
            (shift_register(register_index), new_value)
        };

        rebased.start_fp = new_base;
        rebased.expected_fp = spec.expected_fp.map(|fp| (fp as i64 + delta) as u32);
        rebased.start_registers = spec
            .start_registers
            .iter()
            .map(|&(reg, val)| shift_reg_entry(reg, val))
            .collect();
        rebased.expected_registers = spec
            .expected_registers
            .iter()
            .map(|&(reg, val)| shift_reg_entry(reg, val))
            .collect();
        rebased.fp_value_registers = spec
            .fp_value_registers
            .iter()
            .map(|&reg| shift_register(reg))
            .collect();
        rebased
    }

    fn test_spec_for_all_register_bases(spec: TestSpec) {
        test_spec(rebase_spec(&spec, REG_BASE_FITS_1_BYTE));
        test_spec(rebase_spec(&spec, REG_BASE_FITS_2_BYTES));
        test_spec(rebase_spec(&spec, REG_BASE_FITS_3_BYTES));
    }

    // ==================== BaseAlu Tests ====================

    #[test]
    fn test_add_imm() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![add_imm(1, 0, 100_i16)],
            start_fp: 124,
            expected_registers: vec![(125, 100)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec);
    }

    #[test]
    fn test_add() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] + reg[fp+1]
        // 30 + 12 = 42
        let spec = TestSpec {
            program: vec![add(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 30), (11, 12)],
            expected_registers: vec![(12, 42)],
            ..Default::default()
        };

        // Test all register bases on all backends (CPU + GPU)
        test_spec_for_all_register_bases(spec);
    }

    #[test]
    fn test_add_byte_carry() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] + reg[fp+1]
        // 0xFF + 1 = 0x100 (carry into second byte)
        let spec = TestSpec {
            program: vec![add(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0xFF), (11, 1)],
            expected_registers: vec![(12, 0x100)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec);
    }

    #[test]
    fn test_add_u32_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] + reg[fp+1]
        // 0xFFFFFFFF + 1 = 0 (wrapping overflow)
        let spec = TestSpec {
            program: vec![add(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0xFFFFFFFF), (11, 1)],
            expected_registers: vec![(12, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec);
    }

    #[test]
    fn test_sub() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] - reg[fp+1]
        // 100 - 42 = 58
        let spec = TestSpec {
            program: vec![sub(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 100), (11, 42)],
            expected_registers: vec![(12, 58)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec);
    }

    #[test]
    fn test_sub_negative_result() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] - reg[fp+1]
        // 10 - 20 = -10 (wraps to 0xFFFFFFF6)
        let spec = TestSpec {
            program: vec![sub(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 10), (11, 20)],
            expected_registers: vec![(12, 0xFFFFFFF6)], // -10 as u32
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec);
    }

    #[test]
    fn test_sub_underflow() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] - reg[fp+1]
        // 0 - 1 = 0xFFFFFFFF (wrapping underflow)
        let spec = TestSpec {
            program: vec![sub(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0), (11, 1)],
            expected_registers: vec![(12, 0xFFFFFFFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec);
    }

    #[test]
    fn test_xor() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] ^ reg[fp+1]
        // 0b1010 ^ 0b1100 = 0b0110
        let spec = TestSpec {
            program: vec![xor(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0b1010), (11, 0b1100)],
            expected_registers: vec![(12, 0b0110)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec);
    }

    #[test]
    fn test_or() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] | reg[fp+1]
        // 0b1010 | 0b1100 = 0b1110
        let spec = TestSpec {
            program: vec![or(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0b1010), (11, 0b1100)],
            expected_registers: vec![(12, 0b1110)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec);
    }

    #[test]
    fn test_and() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] & reg[fp+1]
        // 0b1010 & 0b1100 = 0b1000
        let spec = TestSpec {
            program: vec![and(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0b1010), (11, 0b1100)],
            expected_registers: vec![(12, 0b1000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec);
    }

    #[test]
    fn test_and_imm() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = reg[fp+0] & 0xFF
        // 0x1234 & 0xFF = 0x34
        let spec = TestSpec {
            program: vec![and_imm(1, 0, 0xFF_i16)],
            start_fp: 10,
            start_registers: vec![(10, 0x1234)],
            expected_registers: vec![(11, 0x34)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec);
    }

    // ==================== LoadStore and LoadSignExtend Tests ====================

    #[test]
    fn test_loadw() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 0]
        // Load 0xDEADBEEF from address 100
        let spec = TestSpec {
            program: vec![loadw(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(10, 100)], // base address = 100
            start_ram: vec![(100, 0xDEADBEEF)],
            expected_registers: vec![(11, 0xDEADBEEF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_loadw_with_offset() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 8]
        // Load from address 100 + 8 = 108
        let spec = TestSpec {
            program: vec![loadw(1, 0, 8)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            start_ram: vec![(108, 0x12345678)],
            expected_registers: vec![(11, 0x12345678)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_storew() {
        setup_tracing_with_log_level(Level::WARN);

        // MEM[reg[fp+1] + 0] = reg[fp+0]
        // Store 0xCAFEBABE at address 200
        let spec = TestSpec {
            program: vec![storew(0, 1, 0)],
            start_fp: 10,
            start_registers: vec![(10, 0xCAFEBABE), (11, 200)],
            expected_ram: vec![(200, 0xCAFEBABE)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_storew_with_offset() {
        setup_tracing_with_log_level(Level::WARN);

        // MEM[reg[fp+1] + 4] = reg[fp+0]
        // Store at address 200 + 4 = 204
        let spec = TestSpec {
            program: vec![storew(0, 1, 4)],
            start_fp: 10,
            start_registers: vec![(10, 0x11223344), (11, 200)],
            expected_ram: vec![(204, 0x11223344)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_loadbu() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 0] (zero-extended byte)
        // Load byte 0xAB, should remain 0x000000AB
        let spec = TestSpec {
            program: vec![loadbu(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            start_ram: vec![(100, 0xFFFFFFAB)], // Only lowest byte matters
            expected_registers: vec![(11, 0xAB)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_loadhu() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 0] (zero-extended halfword)
        // Load halfword 0xABCD, should remain 0x0000ABCD
        let spec = TestSpec {
            program: vec![loadhu(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            start_ram: vec![(100, 0xFFFFABCD)], // Only lowest 2 bytes matter
            expected_registers: vec![(11, 0xABCD)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_storeb() {
        setup_tracing_with_log_level(Level::WARN);

        // MEM[reg[fp+1] + 0] = reg[fp+0] (lowest byte only)
        // Store byte 0x42 at address 200
        let spec = TestSpec {
            program: vec![storeb(0, 1, 0)],
            start_fp: 10,
            start_registers: vec![(10, 0x12345642), (11, 200)],
            start_ram: vec![(200, 0xFFFFFFFF)], // Prefill with ones
            expected_ram: vec![(200, 0xFFFFFF42)], // Only lowest byte changed
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_storeh() {
        setup_tracing_with_log_level(Level::WARN);

        // MEM[reg[fp+1] + 0] = reg[fp+0] (lowest halfword only)
        // Store halfword 0xBEEF at address 200
        let spec = TestSpec {
            program: vec![storeh(0, 1, 0)],
            start_fp: 10,
            start_registers: vec![(10, 0x1234BEEF), (11, 200)],
            start_ram: vec![(200, 0xFFFFFFFF)], // Prefill with ones
            expected_ram: vec![(200, 0xFFFFBEEF)], // Only lowest 2 bytes changed
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_loadb_positive() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 0] (sign-extended byte)
        // Load byte 0x7F (positive), should remain 0x0000007F
        let spec = TestSpec {
            program: vec![loadb(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            start_ram: vec![(100, 0x0000007F)],
            expected_registers: vec![(11, 0x0000007F)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_loadb_negative() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 0] (sign-extended byte)
        // Load byte 0x80 (negative), should become 0xFFFFFF80
        let spec = TestSpec {
            program: vec![loadb(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            start_ram: vec![(100, 0x00000080)],
            expected_registers: vec![(11, 0xFFFFFF80)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_loadh_positive() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 0] (sign-extended halfword)
        // Load halfword 0x7FFF (positive), should remain 0x00007FFF
        let spec = TestSpec {
            program: vec![loadh(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            start_ram: vec![(100, 0x00007FFF)],
            expected_registers: vec![(11, 0x00007FFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_loadh_negative() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = MEM[reg[fp+0] + 0] (sign-extended halfword)
        // Load halfword 0x8000 (negative), should become 0xFFFF8000
        let spec = TestSpec {
            program: vec![loadh(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            start_ram: vec![(100, 0x00008000)],
            expected_registers: vec![(11, 0xFFFF8000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_storeb_with_offset_roundtrip() {
        setup_tracing_with_log_level(Level::WARN);

        // Store byte at two offsets, load them back and add
        let spec = TestSpec {
            program: vec![
                storeb(0, 1, 0), // MEM[500+0] = lowest byte of reg[0] = 0x34
                storeb(0, 1, 1), // MEM[500+1] = lowest byte of reg[0] = 0x34
                loadbu(2, 1, 0), // reg[2] = 0x34
                loadbu(3, 1, 1), // reg[3] = 0x34
                add(4, 2, 3),    // reg[4] = 0x34 + 0x34 = 104
            ],
            start_fp: 10,
            start_registers: vec![(10, 0x1234), (11, 500)],
            expected_registers: vec![(14, 104)],
            expected_ram: vec![(500, 0x00003434)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_storeh_with_offset_roundtrip() {
        setup_tracing_with_log_level(Level::WARN);

        // Store halfwords at two offsets, load them back and add
        let spec = TestSpec {
            program: vec![
                storeh(0, 2, 0), // MEM[600+0] = 0x1111
                storeh(1, 2, 2), // MEM[600+2] = 0x2222
                loadhu(3, 2, 0), // reg[3] = 0x1111
                loadhu(4, 2, 2), // reg[4] = 0x2222
                add(5, 3, 4),    // reg[5] = 0x1111 + 0x2222 = 13107
            ],
            start_fp: 10,
            start_registers: vec![(10, 0x1111), (11, 0x2222), (12, 600)],
            expected_registers: vec![(15, 13107)],
            expected_ram: vec![(600, 0x22221111)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }
    // ==================== LoadB/LoadH large immediate tests ====================

    #[test]
    fn test_loadb_large_imm() {
        setup_tracing_with_log_level(Level::WARN);

        // Test loadb with a large immediate where the upper 16 bits > 1.
        // This is the exact pattern that was broken: imm = 135015 = 0x20F67,
        // which splits into c = 0x0F67 (lower 16) and g = 0x0002 (upper 16).
        // The bug reconstructed this as c + g * 0xFFFF0000 = wrong address.
        let imm: u32 = 0x2_0004; // 131076, upper 16 bits = 2
        let addr: u32 = 100;
        let target = addr + imm;

        let spec = TestSpec {
            program: vec![loadb(1, 0, imm)],
            start_fp: 10,
            start_registers: vec![(10, addr)],
            start_ram: vec![(target, 0x0000007F)], // positive byte
            expected_registers: vec![(11, 0x0000007F)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_loadb_large_imm_negative() {
        setup_tracing_with_log_level(Level::WARN);

        // Same as above but with a negative byte to verify sign extension
        let imm: u32 = 0x2_0004;
        let addr: u32 = 100;
        let target = addr + imm;

        let spec = TestSpec {
            program: vec![loadb(1, 0, imm)],
            start_fp: 10,
            start_registers: vec![(10, addr)],
            start_ram: vec![(target, 0x000000FE)], // -2 as signed byte
            expected_registers: vec![(11, 0xFFFFFFFE)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_loadh_large_imm() {
        setup_tracing_with_log_level(Level::WARN);

        // Test loadh with a large immediate (upper 16 bits > 1)
        let imm: u32 = 0x3_0000; // 196608, upper 16 bits = 3
        let addr: u32 = 200;
        let target = addr + imm;

        let spec = TestSpec {
            program: vec![loadh(1, 0, imm)],
            start_fp: 10,
            start_registers: vec![(10, addr)],
            start_ram: vec![(target, 0x00008001)], // negative halfword
            expected_registers: vec![(11, 0xFFFF8001)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_storeb_loadb_large_imm_roundtrip() {
        setup_tracing_with_log_level(Level::WARN);

        // Store a byte with a large immediate, then load it back with loadb
        // (sign-extending). This tests that STOREB and LOADB agree on the address
        // when the immediate has upper 16 bits > 1.
        let imm: u32 = 0x2_0004;
        let addr: u32 = 100;
        let target = addr + imm;

        let spec = TestSpec {
            program: vec![
                storeb(0, 1, imm), // MEM[addr + imm] = 0x42
                loadb(2, 1, imm),  // reg[2] = sign_extend(MEM[addr + imm])
            ],
            start_fp: 10,
            start_registers: vec![(10, 0x42), (11, addr)],
            start_ram: vec![(target, 0)],
            expected_registers: vec![(12, 0x42)],
            expected_ram: vec![(target, 0x42)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== LessThan Tests ====================

    #[test]
    fn test_lt_u_true() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = (reg[fp+0] < reg[fp+1]) unsigned
        // 10 < 20 = 1
        let spec = TestSpec {
            program: vec![lt_u(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 10), (11, 20)],
            expected_registers: vec![(12, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_u_false() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = (reg[fp+0] < reg[fp+1]) unsigned
        // 20 < 10 = 0
        let spec = TestSpec {
            program: vec![lt_u(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 20), (11, 10)],
            expected_registers: vec![(12, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_u_equal() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = (reg[fp+0] < reg[fp+1]) unsigned
        // 42 < 42 = 0
        let spec = TestSpec {
            program: vec![lt_u(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 42), (11, 42)],
            expected_registers: vec![(12, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_s_negative() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = (reg[fp+0] < reg[fp+1]) signed
        // -1 (0xFFFFFFFF) < 1 = 1 (signed)
        let spec = TestSpec {
            program: vec![lt_s(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0xFFFFFFFF), (11, 1)],
            expected_registers: vec![(12, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_u_imm() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = (reg[fp+0] < 100) unsigned
        // 50 < 100 = 1
        let spec = TestSpec {
            program: vec![lt_u_imm(1, 0, 100_i16)],
            start_fp: 10,
            start_registers: vec![(10, 50)],
            expected_registers: vec![(11, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_gt_u_true() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![gt_u(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 20), (11, 10)],
            expected_registers: vec![(12, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_gt_s_negative() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![gt_s(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 1), (11, 0xFFFF_FFFF)],
            expected_registers: vec![(12, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_s_positive() {
        setup_tracing_with_log_level(Level::WARN);

        // 50 < 100 = 1 (signed, both positive)
        let spec = TestSpec {
            program: vec![lt_s(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 50), (11, 100)],
            expected_registers: vec![(12, 1)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_s_both_negative() {
        setup_tracing_with_log_level(Level::WARN);

        // -2 < -4 = 0 (signed, -2 is greater than -4)
        let spec = TestSpec {
            program: vec![lt_s(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, (-2_i32) as u32), (11, (-4_i32) as u32)],
            expected_registers: vec![(12, 0)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_gt_u_false() {
        setup_tracing_with_log_level(Level::WARN);

        // 100 > 200 = 0 (unsigned)
        let spec = TestSpec {
            program: vec![gt_u(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 100), (11, 200)],
            expected_registers: vec![(12, 0)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_gt_u_equal() {
        setup_tracing_with_log_level(Level::WARN);

        // 150 > 150 = 0 (unsigned)
        let spec = TestSpec {
            program: vec![gt_u(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 150), (11, 150)],
            expected_registers: vec![(12, 0)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_gt_s_positive() {
        setup_tracing_with_log_level(Level::WARN);

        // 100 > 50 = 1 (signed)
        let spec = TestSpec {
            program: vec![gt_s(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 100), (11, 50)],
            expected_registers: vec![(12, 1)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_gt_s_both_negative() {
        setup_tracing_with_log_level(Level::WARN);

        // -2 > -4 = 1 (signed)
        let spec = TestSpec {
            program: vec![gt_s(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, (-2_i32) as u32), (11, (-4_i32) as u32)],
            expected_registers: vec![(12, 1)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_comparison_chain() {
        setup_tracing_with_log_level(Level::WARN);

        // (10 < 20) = 1, (20 < 30) = 1
        let spec = TestSpec {
            program: vec![
                lt_u(3, 0, 1), // reg[3] = (10 < 20) = 1
                lt_u(4, 1, 2), // reg[4] = (20 < 30) = 1
            ],
            start_fp: 10,
            start_registers: vec![(10, 10), (11, 20), (12, 30)],
            expected_registers: vec![(13, 1), (14, 1)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_gt_edge_cases() {
        setup_tracing_with_log_level(Level::WARN);

        // max u32 > 0 (unsigned), max positive i32 > 0 (signed)
        let spec = TestSpec {
            program: vec![
                gt_u(2, 0, 1), // reg[2] = (0xFFFFFFFF > 0) unsigned = 1
                gt_s(5, 3, 4), // reg[5] = (0x7FFFFFFF > 0) signed = 1
            ],
            start_fp: 10,
            start_registers: vec![
                (10, 0xFFFFFFFF), // max u32
                (11, 0),
                (13, 0x7FFFFFFF), // max positive i32
                (14, 0),
            ],
            expected_registers: vec![(12, 1), (15, 1)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_comparison_equivalence() {
        setup_tracing_with_log_level(Level::WARN);

        // gt_u(a, b) == lt_u(b, a): xor should be 0
        let spec = TestSpec {
            program: vec![
                gt_u(2, 0, 1), // reg[2] = (25 > 10) = 1
                lt_u(3, 1, 0), // reg[3] = (10 < 25) = 1
                xor(4, 2, 3),  // reg[4] = 0
            ],
            start_fp: 10,
            start_registers: vec![(10, 25), (11, 10)],
            expected_registers: vec![(14, 0)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_mixed_signed_unsigned() {
        setup_tracing_with_log_level(Level::WARN);

        // 0xFFFFFFFE: large unsigned, but -2 signed
        // unsigned: 0xFFFFFFFE > 2 = 1
        // signed: -2 > 2 = 0
        // difference = 1
        let spec = TestSpec {
            program: vec![
                gt_u(2, 0, 1), // reg[2] = (0xFFFFFFFE > 2) unsigned = 1
                gt_s(3, 0, 1), // reg[3] = (-2 > 2) signed = 0
                sub(4, 2, 3),  // reg[4] = 1 - 0 = 1
            ],
            start_fp: 10,
            start_registers: vec![(10, 0xFFFFFFFE), (11, 2)],
            expected_registers: vec![(14, 1)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }
    // ==================== LessThan64 Tests ====================

    #[test]
    fn test_lt_u_64_true() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_0000 < 0x0000_0002_0000_0000 = 1 (unsigned)
        let spec = TestSpec {
            program: vec![lt_u_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 1), // reg 0 = 0x0000_0001_0000_0000
                (126, 0),
                (127, 2), // reg 2 = 0x0000_0002_0000_0000
            ],
            expected_registers: vec![(128, 1), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_u_64_false() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0002_0000_0000 < 0x0000_0001_0000_0000 = 0 (unsigned)
        let spec = TestSpec {
            program: vec![lt_u_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 2), // reg 0 = 0x0000_0002_0000_0000
                (126, 0),
                (127, 1), // reg 2 = 0x0000_0001_0000_0000
            ],
            expected_registers: vec![(128, 0), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_s_64_negative() {
        setup_tracing_with_log_level(Level::WARN);

        // -1 (0xFFFF_FFFF_FFFF_FFFF) < 1 (0x0000_0000_0000_0001) = 1 (signed)
        let spec = TestSpec {
            program: vec![lt_s_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0xFFFF_FFFF),
                (125, 0xFFFF_FFFF), // reg 0 = -1
                (126, 1),
                (127, 0), // reg 2 = 1
            ],
            expected_registers: vec![(128, 1), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_gt_u_64_true() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![gt_u_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![(124, 0), (125, 2), (126, 0), (127, 1)],
            expected_registers: vec![(128, 1), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_gt_s_64_true() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![gt_s_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![(124, 1), (125, 0), (126, 0xFFFF_FFFF), (127, 0xFFFF_FFFF)],
            expected_registers: vec![(128, 1), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== LessThan64 output width tests ====================
    // WASM comparison instructions produce i32 results, so LessThan64 must
    // write only 1 word (32 bits). These tests pre-fill the high word of the
    // destination register and verify it is preserved after the comparison.

    #[test]
    fn test_lt_u_64_preserves_rd_high_word() {
        setup_tracing_with_log_level(Level::WARN);

        // Pre-fill the high word of the destination register (rd=4, high word at fp+5)
        // with 0xDEAD_BEEF. After the comparison, it should be preserved because
        // a comparison result is only 32 bits (i32 in WASM).
        //
        // 0x0000_0001_0000_0000 < 0x0000_0002_0000_0000 = 1 (unsigned)
        let spec = TestSpec {
            program: vec![lt_u_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 1), // reg 0 = 0x0000_0001_0000_0000
                (126, 0),
                (127, 2),           // reg 2 = 0x0000_0002_0000_0000
                (129, 0xDEAD_BEEF), // Pre-fill high word of rd
            ],
            // Result=1 in low word; high word should remain 0xDEAD_BEEF
            expected_registers: vec![(128, 1), (129, 0xDEAD_BEEF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_s_64_preserves_rd_high_word() {
        setup_tracing_with_log_level(Level::WARN);

        // Same idea: pre-fill high word of rd with non-zero data, run signed comparison,
        // verify high word is preserved.
        //
        // -1 (0xFFFF_FFFF_FFFF_FFFF) < 1 (0x0000_0000_0000_0001) = 1 (signed)
        let spec = TestSpec {
            program: vec![lt_s_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0xFFFF_FFFF),
                (125, 0xFFFF_FFFF), // reg 0 = -1
                (126, 1),
                (127, 0),           // reg 2 = 1
                (129, 0xCAFE_BABE), // Pre-fill high word of rd
            ],
            // Result=1 in low word; high word should remain 0xCAFE_BABE
            expected_registers: vec![(128, 1), (129, 0xCAFE_BABE)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_lt_u_64_false_preserves_rd_high_word() {
        setup_tracing_with_log_level(Level::WARN);

        // Even when the result is 0 (false), the high word should not be touched.
        //
        // 0x0000_0002_0000_0000 < 0x0000_0001_0000_0000 = 0 (unsigned)
        let spec = TestSpec {
            program: vec![lt_u_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 2), // reg 0 = 0x0000_0002_0000_0000
                (126, 0),
                (127, 1),           // reg 2 = 0x0000_0001_0000_0000
                (129, 0x1234_5678), // Pre-fill high word of rd
            ],
            // Result=0 in low word; high word should remain 0x1234_5678
            expected_registers: vec![(128, 0), (129, 0x1234_5678)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== Eq Tests ====================

    #[test]
    fn test_eq_true() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = (reg[fp+0] == reg[fp+1])
        // 42 == 42 = 1
        let spec = TestSpec {
            program: vec![eq(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 42), (11, 42)],
            expected_registers: vec![(12, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_eq_false() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = (reg[fp+0] == reg[fp+1])
        // 10 == 20 = 0
        let spec = TestSpec {
            program: vec![eq(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 10), (11, 20)],
            expected_registers: vec![(12, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_neq_true() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = (reg[fp+0] != reg[fp+1])
        // 10 != 20 = 1
        let spec = TestSpec {
            program: vec![neq(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 10), (11, 20)],
            expected_registers: vec![(12, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_neq_false() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = (reg[fp+0] != reg[fp+1])
        // 42 != 42 = 0
        let spec = TestSpec {
            program: vec![neq(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 42), (11, 42)],
            expected_registers: vec![(12, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_eq_imm() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = (reg[fp+0] == 100)
        // 100 == 100 = 1
        let spec = TestSpec {
            program: vec![eq_imm(1, 0, 100_i16)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            expected_registers: vec![(11, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_neq_imm() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = (reg[fp+0] != 100)
        // 50 != 100 = 1
        let spec = TestSpec {
            program: vec![neq_imm(1, 0, 100_i16)],
            start_fp: 10,
            start_registers: vec![(10, 50)],
            expected_registers: vec![(11, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== Eq64 Tests ====================

    #[test]
    fn test_eq_64_true() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_0000 == 0x0000_0001_0000_0000 = 1
        let spec = TestSpec {
            program: vec![eq_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 1), // reg 0 = 0x0000_0001_0000_0000
                (126, 0),
                (127, 1), // reg 2 = 0x0000_0001_0000_0000
            ],
            expected_registers: vec![(128, 1), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_eq_64_false() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_0000 == 0x0000_0002_0000_0000 = 0
        let spec = TestSpec {
            program: vec![eq_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 1), // reg 0 = 0x0000_0001_0000_0000
                (126, 0),
                (127, 2), // reg 2 = 0x0000_0002_0000_0000
            ],
            expected_registers: vec![(128, 0), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_neq_64_true() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_0000 != 0x0000_0002_0000_0000 = 1
        let spec = TestSpec {
            program: vec![neq_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 1), // reg 0 = 0x0000_0001_0000_0000
                (126, 0),
                (127, 2), // reg 2 = 0x0000_0002_0000_0000
            ],
            expected_registers: vec![(128, 1), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_neq_64_false() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_0000 != 0x0000_0001_0000_0000 = 0
        let spec = TestSpec {
            program: vec![neq_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 1), // reg 0 = 0x0000_0001_0000_0000
                (126, 0),
                (127, 1), // reg 2 = 0x0000_0001_0000_0000
            ],
            expected_registers: vec![(128, 0), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== Eq64 output width tests ====================
    // WASM comparison instructions produce i32 results, so Eq64 must
    // write only 1 word (32 bits). These tests pre-fill the high word of the
    // destination register and verify it is preserved after the comparison.

    #[test]
    fn test_eq_64_preserves_rd_high_word() {
        setup_tracing_with_log_level(Level::WARN);

        // Pre-fill the high word of the destination register (rd=4, high word at fp+5)
        // with 0xDEAD_BEEF. After the comparison, it should be preserved.
        //
        // 0x0000_0001_0000_0000 == 0x0000_0001_0000_0000 = 1
        let spec = TestSpec {
            program: vec![eq_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 1), // reg 0 = 0x0000_0001_0000_0000
                (126, 0),
                (127, 1),           // reg 2 = 0x0000_0001_0000_0000
                (129, 0xDEAD_BEEF), // Pre-fill high word of rd
            ],
            // Result=1 in low word; high word should remain 0xDEAD_BEEF
            expected_registers: vec![(128, 1), (129, 0xDEAD_BEEF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_neq_64_preserves_rd_high_word() {
        setup_tracing_with_log_level(Level::WARN);

        // Same idea: pre-fill high word of rd, run neq, verify high word is preserved.
        //
        // 0x0000_0001_0000_0000 != 0x0000_0002_0000_0000 = 1
        let spec = TestSpec {
            program: vec![neq_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 1), // reg 0 = 0x0000_0001_0000_0000
                (126, 0),
                (127, 2),           // reg 2 = 0x0000_0002_0000_0000
                (129, 0xCAFE_BABE), // Pre-fill high word of rd
            ],
            // Result=1 in low word; high word should remain 0xCAFE_BABE
            expected_registers: vec![(128, 1), (129, 0xCAFE_BABE)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_eq_64_false_preserves_rd_high_word() {
        setup_tracing_with_log_level(Level::WARN);

        // Even when the result is 0 (false), the high word should not be touched.
        //
        // 0x0000_0001_0000_0000 == 0x0000_0002_0000_0000 = 0
        let spec = TestSpec {
            program: vec![eq_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 1), // reg 0 = 0x0000_0001_0000_0000
                (126, 0),
                (127, 2),           // reg 2 = 0x0000_0002_0000_0000
                (129, 0x1234_5678), // Pre-fill high word of rd
            ],
            // Result=0 in low word; high word should remain 0x1234_5678
            expected_registers: vec![(128, 0), (129, 0x1234_5678)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_eq_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = (reg[fp+0] == 42)  (64-bit comparison, imm sign-extended)
        // 42 == 42 → 1
        let spec = TestSpec {
            program: vec![eq_imm_64(2, 0, 42_i16)],
            start_fp: 124,
            start_registers: vec![
                (124, 42),
                (125, 0), // reg 0 = 42 (fits in low word)
            ],
            expected_registers: vec![(126, 1), (127, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_neq_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = (reg[fp+0] != 42)  (64-bit comparison, imm sign-extended)
        // 99 != 42 → 1
        let spec = TestSpec {
            program: vec![neq_imm_64(2, 0, 42_i16)],
            start_fp: 124,
            start_registers: vec![
                (124, 99),
                (125, 0), // reg 0 = 99
            ],
            expected_registers: vec![(126, 1), (127, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== Shift Tests ====================

    #[test]
    fn test_shl() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] << reg[fp+1]
        // 0x01 << 4 = 0x10
        let spec = TestSpec {
            program: vec![shl(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0x01), (11, 4)],
            expected_registers: vec![(12, 0x10)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_shl_imm() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = reg[fp+0] << 8
        // 0xFF << 8 = 0xFF00
        let spec = TestSpec {
            program: vec![shl_imm(1, 0, 8_i16)],
            start_fp: 10,
            start_registers: vec![(10, 0xFF)],
            expected_registers: vec![(11, 0xFF00)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_shr_u() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] >> reg[fp+1] (logical)
        // 0x80000000 >> 4 = 0x08000000
        let spec = TestSpec {
            program: vec![shr_u(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0x80000000), (11, 4)],
            expected_registers: vec![(12, 0x08000000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_shr_u_imm() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![shr_u_imm(1, 0, 4_i16)],
            start_fp: 10,
            start_registers: vec![(10, 0x8000_0000)],
            expected_registers: vec![(11, 0x0800_0000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_shr_s_imm() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = reg[fp+0] >> 4 (arithmetic)
        // 0x80000000 >> 4 = 0xF8000000 (sign-extended)
        let spec = TestSpec {
            program: vec![shr_s_imm(1, 0, 4_i16)],
            start_fp: 10,
            start_registers: vec![(10, 0x80000000)],
            expected_registers: vec![(11, 0xF8000000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_shr_s_imm_positive() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = reg[fp+0] >> 8 (arithmetic, positive value)
        // 0x7FFF0000 >> 8 = 0x007FFF00 (no sign extension since MSB is 0)
        let spec = TestSpec {
            program: vec![shr_s_imm(1, 0, 8_i16)],
            start_fp: 10,
            start_registers: vec![(10, 0x7FFF0000)],
            expected_registers: vec![(11, 0x007FFF00)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_shl_large_shift() {
        setup_tracing_with_log_level(Level::WARN);

        // Shift by 31 bits
        // 0x01 << 31 = 0x80000000
        let spec = TestSpec {
            program: vec![shl(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0x01), (11, 31)],
            expected_registers: vec![(12, 0x80000000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== Shift64 Tests ====================

    #[test]
    fn test_shl_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_0000 << 4 = 0x0000_0010_0000_0000
        let spec = TestSpec {
            program: vec![shl_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 1), // reg 0 = 0x0000_0001_0000_0000
                (126, 4),
                (127, 0), // reg 2 = 4
            ],
            expected_registers: vec![(128, 0), (129, 0x10)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_shl_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![shl_imm_64(2, 0, 8_i16)],
            start_fp: 124,
            start_registers: vec![(124, 0), (125, 1)],
            expected_registers: vec![(126, 0), (127, 0x0000_0100)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_shr_u_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x8000_0000_0000_0000 >> 4 = 0x0800_0000_0000_0000
        let spec = TestSpec {
            program: vec![shr_u_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 0x80000000), // reg 0 = 0x8000_0000_0000_0000
                (126, 4),
                (127, 0), // reg 2 = 4
            ],
            expected_registers: vec![(128, 0), (129, 0x08000000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_shr_u_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![shr_u_imm_64(2, 0, 8_i16)],
            start_fp: 124,
            start_registers: vec![(124, 0), (125, 0x8000_0000)],
            expected_registers: vec![(126, 0), (127, 0x0080_0000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_shr_s_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x8000_0000_0000_0000 >> 4 (arithmetic) = 0xF800_0000_0000_0000
        let spec = TestSpec {
            program: vec![shr_s_imm_64(2, 0, 4_i16)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 0x80000000), // reg 0 = 0x8000_0000_0000_0000
            ],
            expected_registers: vec![(126, 0), (127, 0xF8000000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_shr_s_imm_64_positive() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x7FFF_0000_0000_0000 >> 8 (arithmetic) = 0x007F_FF00_0000_0000
        // No sign extension since MSB is 0
        let spec = TestSpec {
            program: vec![shr_s_imm_64(2, 0, 8_i16)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 0x7FFF0000), // reg 0 = 0x7FFF_0000_0000_0000
            ],
            expected_registers: vec![(126, 0), (127, 0x007FFF00)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== add_64 ====================

    #[test]
    fn test_add_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_ffff_0000_0001 + 0x80 = 0x0000_ffff_0000_0081
        let spec = TestSpec {
            program: vec![add_imm_64(2, 0, 0x80_i16)],
            start_fp: 124,
            start_registers: vec![(124, 1), (125, 0xffff)],
            expected_registers: vec![(126, 0x81), (127, 0xffff)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_add_imm_64_low_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_FFFF_FF00 + 0x0100 = 0x0000_0002_0000_0000
        // Low limb overflows, carry propagates to high limb
        let spec = TestSpec {
            program: vec![add_imm_64(2, 0, 0x0100_i16)],
            start_fp: 124,
            start_registers: vec![(124, 0xFFFF_FF00), (125, 0x0000_0001)],
            expected_registers: vec![(126, 0x0000_0000), (127, 0x0000_0002)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_add_imm_64_full_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // 0xFFFF_FFFF_FFFF_FFFF + 1 = 0 (wraps around)
        let spec = TestSpec {
            program: vec![add_imm_64(2, 0, 1_i16)],
            start_fp: 124,
            start_registers: vec![(124, 0xFFFF_FFFF), (125, 0xFFFF_FFFF)],
            expected_registers: vec![(126, 0), (127, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_add_64_reg() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_0003 + 0x0000_0002_0000_0004 = 0x0000_0003_0000_0007
        let spec = TestSpec {
            program: vec![add_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 3),
                (125, 1), // reg 0 = 0x0000_0001_0000_0003
                (126, 4),
                (127, 2), // reg 2 = 0x0000_0002_0000_0004
            ],
            expected_registers: vec![(128, 7), (129, 3)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_add_64_reg_low_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_8000_0000 + 0x0000_0001_8000_0000 = 0x0000_0003_0000_0000
        let spec = TestSpec {
            program: vec![add_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0x8000_0000),
                (125, 1), // reg 0
                (126, 0x8000_0000),
                (127, 1), // reg 2
            ],
            expected_registers: vec![(128, 0x0000_0000), (129, 3)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_add_64_reg_full_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // 0xFFFF_FFFF_FFFF_FFFE + 0x0000_0000_0000_0003 = 0x0000_0000_0000_0001
        let spec = TestSpec {
            program: vec![add_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0xFFFF_FFFE),
                (125, 0xFFFF_FFFF), // reg 0
                (126, 3),
                (127, 0), // reg 2
            ],
            expected_registers: vec![(128, 1), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== sub_64 ====================

    #[test]
    fn test_sub_64_reg() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0003_0000_0007 - 0x0000_0001_0000_0003 = 0x0000_0002_0000_0004
        let spec = TestSpec {
            program: vec![sub_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 7),
                (125, 3), // reg 0
                (126, 3),
                (127, 1), // reg 2
            ],
            expected_registers: vec![(128, 4), (129, 2)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_sub_64_reg_low_borrow() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0003_0000_0000 - 0x0000_0001_0000_0001 = 0x0000_0001_FFFF_FFFF
        // Low limb borrows from high limb
        let spec = TestSpec {
            program: vec![sub_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0x0000_0000),
                (125, 3), // reg 0
                (126, 1),
                (127, 1), // reg 2
            ],
            expected_registers: vec![(128, 0xFFFF_FFFF), (129, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_sub_64_reg_full_underflow() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0000_0000_0001 - 0x0000_0000_0000_0003 = 0xFFFF_FFFF_FFFF_FFFE (wraps)
        let spec = TestSpec {
            program: vec![sub_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 1),
                (125, 0), // reg 0
                (126, 3),
                (127, 0), // reg 2
            ],
            expected_registers: vec![(128, 0xFFFF_FFFE), (129, 0xFFFF_FFFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_sub_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0002_0000_0000 - 1 = 0x0000_0001_FFFF_FFFF
        // Low borrow propagates to high limb
        let spec = TestSpec {
            program: vec![sub_imm_64(2, 0, 1_i16)],
            start_fp: 124,
            start_registers: vec![(124, 0x0000_0000), (125, 2)],
            expected_registers: vec![(126, 0xFFFF_FFFF), (127, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== xor_64 ====================

    #[test]
    fn test_xor_64_reg() {
        setup_tracing_with_log_level(Level::WARN);

        // 0xDEAD_BEEF_CAFE_BABE ^ 0xFFFF_FFFF_0000_0000 = 0x2152_4110_CAFE_BABE
        let spec = TestSpec {
            program: vec![xor_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0xCAFE_BABE),
                (125, 0xDEAD_BEEF), // reg 0
                (126, 0x0000_0000),
                (127, 0xFFFF_FFFF), // reg 2
            ],
            expected_registers: vec![(128, 0xCAFE_BABE), (129, 0x2152_4110)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_xor_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_00FF ^ 0xFF (sign-extended to 0x0000_0000_0000_00FF) = 0x0000_0001_0000_0000
        let spec = TestSpec {
            program: vec![xor_imm_64(2, 0, 0xFF_i16)],
            start_fp: 124,
            start_registers: vec![(124, 0x0000_00FF), (125, 1)],
            expected_registers: vec![(126, 0x0000_0000), (127, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== or_64 ====================

    #[test]
    fn test_or_64_reg() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x00FF_00FF_00FF_00FF | 0xFF00_FF00_FF00_FF00 = 0xFFFF_FFFF_FFFF_FFFF
        let spec = TestSpec {
            program: vec![or_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0x00FF_00FF),
                (125, 0x00FF_00FF), // reg 0
                (126, 0xFF00_FF00),
                (127, 0xFF00_FF00), // reg 2
            ],
            expected_registers: vec![(128, 0xFFFF_FFFF), (129, 0xFFFF_FFFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_or_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_0000 | 0x0F (sign-extended to 0x0000_0000_0000_000F) = 0x0000_0001_0000_000F
        let spec = TestSpec {
            program: vec![or_imm_64(2, 0, 0x0F_i16)],
            start_fp: 124,
            start_registers: vec![(124, 0x0000_0000), (125, 1)],
            expected_registers: vec![(126, 0x0000_000F), (127, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== and_64 ====================

    #[test]
    fn test_and_64_reg() {
        setup_tracing_with_log_level(Level::WARN);

        // 0xFFFF_0000_FFFF_0000 & 0x0F0F_0F0F_0F0F_0F0F = 0x0F0F_0000_0F0F_0000
        let spec = TestSpec {
            program: vec![and_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0xFFFF_0000),
                (125, 0xFFFF_0000), // reg 0
                (126, 0x0F0F_0F0F),
                (127, 0x0F0F_0F0F), // reg 2
            ],
            expected_registers: vec![(128, 0x0F0F_0000), (129, 0x0F0F_0000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_and_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0xDEAD_BEEF_CAFE_BABE & 0xFF (sign-extended to 0x0000_0000_0000_00FF) = 0x0000_0000_0000_00BE
        let spec = TestSpec {
            program: vec![and_imm_64(2, 0, 0xFF_i16)],
            start_fp: 124,
            start_registers: vec![(124, 0xCAFE_BABE), (125, 0xDEAD_BEEF)],
            expected_registers: vec![(126, 0x0000_00BE), (127, 0x0000_0000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== Mul Tests ====================

    #[test]
    fn test_mul() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+2] = reg[fp+0] * reg[fp+1]
        // 7 * 6 = 42
        let spec = TestSpec {
            program: vec![mul(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 7), (11, 6)],
            expected_registers: vec![(12, 42)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_mul_imm() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = reg[fp+0] * 10
        // 42 * 10 = 420
        let spec = TestSpec {
            program: vec![mul_imm(1, 0, 10_i16)],
            start_fp: 10,
            start_registers: vec![(10, 42)],
            expected_registers: vec![(11, 420)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_mul_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x10000 * 0x10000 = 0x1_0000_0000, wraps to 0
        let spec = TestSpec {
            program: vec![mul(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0x10000), (11, 0x10000)],
            expected_registers: vec![(12, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_mul_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0000_0000_0007 * 0x0000_0000_0000_0006 = 0x0000_0000_0000_002A
        let spec = TestSpec {
            program: vec![mul_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 7),
                (125, 0), // reg 0 = 7
                (126, 6),
                (127, 0), // reg 2 = 6
            ],
            expected_registers: vec![(128, 0x2a), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec);
    }

    #[test]
    fn test_mul_64_large() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_0000 * 0x0000_0000_0000_0003 = 0x0000_0003_0000_0000
        let spec = TestSpec {
            program: vec![mul_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 1), // reg 0 = 0x1_0000_0000
                (126, 3),
                (127, 0), // reg 2 = 3
            ],
            expected_registers: vec![(128, 0), (129, 3)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec);
    }

    #[test]
    fn test_mul_64_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_0001 * 0x0000_0001_0000_0001 = wraps to 0x0000_0002_0000_0001
        let spec = TestSpec {
            program: vec![mul_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 1),
                (125, 1), // reg 0 = 0x1_0000_0001
                (126, 1),
                (127, 1), // reg 2 = 0x1_0000_0001
            ],
            expected_registers: vec![(128, 1), (129, 2)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec);
    }

    #[test]
    fn test_mul_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0000_0000_000A * 3 = 0x0000_0000_0000_001E
        let spec = TestSpec {
            program: vec![mul_imm_64(2, 0, 3_i16.into())],
            start_fp: 124,
            start_registers: vec![(124, 0xa), (125, 0)],
            expected_registers: vec![(126, 0x1e), (127, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec);
    }

    #[test]
    fn test_mul_zero() {
        setup_tracing_with_log_level(Level::WARN);

        // 12345 * 0 = 0
        let spec = TestSpec {
            program: vec![mul(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 12345), (11, 0)],
            expected_registers: vec![(12, 0)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_mul_one() {
        setup_tracing_with_log_level(Level::WARN);

        // 999 * 1 = 999
        let spec = TestSpec {
            program: vec![mul(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 999), (11, 1)],
            expected_registers: vec![(12, 999)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_mul_large_numbers() {
        setup_tracing_with_log_level(Level::WARN);

        // 65537 * 65521 = 4294049777
        let spec = TestSpec {
            program: vec![mul(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 65537), (11, 65521)],
            expected_registers: vec![(12, 4294049777u32)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_mul_max_value() {
        setup_tracing_with_log_level(Level::WARN);

        // 0xFFFFFFFF * 1 = 0xFFFFFFFF
        let spec = TestSpec {
            program: vec![mul(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0xFFFFFFFF), (11, 1)],
            expected_registers: vec![(12, 0xFFFFFFFF)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_mul_negative_positive() {
        setup_tracing_with_log_level(Level::WARN);

        // (-5) * 3 = -15
        let spec = TestSpec {
            program: vec![mul(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, (-5_i32) as u32), (11, 3)],
            expected_registers: vec![(12, (-15_i32) as u32)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_mul_positive_negative() {
        setup_tracing_with_log_level(Level::WARN);

        // 4 * (-6) = -24
        let spec = TestSpec {
            program: vec![mul(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 4), (11, (-6_i32) as u32)],
            expected_registers: vec![(12, (-24_i32) as u32)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_mul_both_negative() {
        setup_tracing_with_log_level(Level::WARN);

        // (-7) * (-3) = 21
        let spec = TestSpec {
            program: vec![mul(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, (-7_i32) as u32), (11, (-3_i32) as u32)],
            expected_registers: vec![(12, 21)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_mul_negative_one() {
        setup_tracing_with_log_level(Level::WARN);

        // 42 * (-1) = -42
        let spec = TestSpec {
            program: vec![mul(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 42), (11, (-1_i32) as u32)],
            expected_registers: vec![(12, (-42_i32) as u32)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_mul_negative_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // INT32_MIN * (-1) = INT32_MIN (0x80000000) due to overflow
        let spec = TestSpec {
            program: vec![mul(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0x80000000), (11, (-1_i32) as u32)],
            expected_registers: vec![(12, 0x80000000)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_mul_commutative() {
        setup_tracing_with_log_level(Level::WARN);

        // Verify a*b == b*a: (13*17) - (17*13) = 0
        let spec = TestSpec {
            program: vec![
                mul(2, 0, 1), // reg[2] = 13 * 17
                mul(3, 1, 0), // reg[3] = 17 * 13
                sub(4, 2, 3), // reg[4] = 0
            ],
            start_fp: 10,
            start_registers: vec![(10, 13), (11, 17)],
            expected_registers: vec![(14, 0)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_mul_chain() {
        setup_tracing_with_log_level(Level::WARN);

        // 2 * 3 = 6, 6 * 5 = 30
        let spec = TestSpec {
            program: vec![
                mul(3, 0, 1), // reg[3] = 2 * 3 = 6
                mul(4, 3, 2), // reg[4] = 6 * 5 = 30
            ],
            start_fp: 10,
            start_registers: vec![(10, 2), (11, 3), (12, 5)],
            expected_registers: vec![(14, 30)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    // ==================== Jump Tests ====================

    #[test]
    fn test_jump() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![
                jump(8),
                halt(), // Should be skipped!
            ],
            expected_pc: Some(8),
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_jump_if_true() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![
                jump_if(2, 8),
                halt(), // Should be skipped!
            ],
            start_fp: 10,
            start_registers: vec![(12, 5)], // Should jump
            expected_pc: Some(8),
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_jump_if_false() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![jump_if(2, 8)],
            start_fp: 10,
            start_registers: vec![(12, 0)], // Should not jump
            expected_pc: Some(4),
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_jump_if_zero_true() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![
                jump_if_zero(2, 8),
                halt(), // Should be skipped!
            ],
            start_fp: 10,
            start_registers: vec![(12, 0)], // Should jump
            expected_pc: Some(8),
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_jump_if_zero_false() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![jump_if_zero(2, 2)],
            start_fp: 10,
            start_registers: vec![(12, 5)], // Should not jump
            expected_pc: Some(4),
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_skip() {
        setup_tracing_with_log_level(Level::WARN);

        let spec = TestSpec {
            program: vec![
                skip(2),
                halt(), // Should be skipped!
            ],
            start_fp: 10,
            start_registers: vec![(12, 1)],
            expected_pc: Some(8),
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== Const32 Tests ====================

    #[test]
    fn test_const32_small() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = 42
        let spec = TestSpec {
            program: vec![const_32_imm(1, 42, 0)],
            start_fp: 10,
            expected_registers: vec![(11, 42)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_const32_large() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = 0xDEADBEEF
        // imm_lo = 0xBEEF, imm_hi = 0xDEAD
        let spec = TestSpec {
            program: vec![const_32_imm(1, 0xBEEF, 0xDEAD)],
            start_fp: 10,
            expected_registers: vec![(11, 0xDEADBEEF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_const32_zero() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = 0
        let spec = TestSpec {
            program: vec![const_32_imm(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(11, 0x12345678)], // Should be overwritten
            expected_registers: vec![(11, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_const32_max() {
        setup_tracing_with_log_level(Level::WARN);

        // reg[fp+1] = 0xFFFFFFFF
        let spec = TestSpec {
            program: vec![const_32_imm(1, 0xFFFF, 0xFFFF)],
            start_fp: 10,
            expected_registers: vec![(11, 0xFFFFFFFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== DivRem Tests ====================

    #[test]
    fn test_div() {
        setup_tracing_with_log_level(Level::WARN);

        // 42 / 7 = 6
        let spec = TestSpec {
            program: vec![div(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 42), (11, 7)],
            expected_registers: vec![(12, 6)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_signed() {
        setup_tracing_with_log_level(Level::WARN);

        // -42 / 7 = -6
        let spec = TestSpec {
            program: vec![div(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, (-42_i32) as u32), (11, 7)],
            expected_registers: vec![(12, (-6_i32) as u32)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_by_zero() {
        setup_tracing_with_log_level(Level::WARN);

        // 42 / 0 = 0xFFFFFFFF (RISC-V spec)
        let spec = TestSpec {
            program: vec![div(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 42), (11, 0)],
            expected_registers: vec![(12, 0xFFFFFFFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_divu() {
        setup_tracing_with_log_level(Level::WARN);

        // 100 / 7 = 14
        let spec = TestSpec {
            program: vec![divu(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 100), (11, 7)],
            expected_registers: vec![(12, 14)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_divu_by_zero() {
        setup_tracing_with_log_level(Level::WARN);

        // 42 / 0 = 0xFFFFFFFF (RISC-V spec)
        let spec = TestSpec {
            program: vec![divu(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 42), (11, 0)],
            expected_registers: vec![(12, 0xFFFF_FFFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_rems() {
        setup_tracing_with_log_level(Level::WARN);

        // 42 % 7 = 0
        // 43 % 7 = 1
        let spec = TestSpec {
            program: vec![rems(2, 0, 1), rems(5, 3, 4)],
            start_fp: 10,
            start_registers: vec![(10, 42), (11, 7), (13, 43), (14, 7)],
            expected_registers: vec![(12, 0), (15, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_rems_negative() {
        setup_tracing_with_log_level(Level::WARN);

        // -43 % 7 = -1 (remainder has sign of dividend)
        let spec = TestSpec {
            program: vec![rems(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, (-43_i32) as u32), (11, 7)],
            expected_registers: vec![(12, (-1_i32) as u32)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_rems_by_zero() {
        setup_tracing_with_log_level(Level::WARN);

        // 42 % 0 = 42 (RISC-V spec: returns dividend)
        let spec = TestSpec {
            program: vec![rems(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 42), (11, 0)],
            expected_registers: vec![(12, 42)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_remu() {
        setup_tracing_with_log_level(Level::WARN);

        // 100 % 7 = 2
        let spec = TestSpec {
            program: vec![remu(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 100), (11, 7)],
            expected_registers: vec![(12, 2)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_remu_by_zero() {
        setup_tracing_with_log_level(Level::WARN);

        // 42 % 0 = 42 (RISC-V spec: returns dividend)
        let spec = TestSpec {
            program: vec![remu(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 42), (11, 0)],
            expected_registers: vec![(12, 42)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_by_one() {
        setup_tracing_with_log_level(Level::WARN);

        // 999 / 1 = 999
        let spec = TestSpec {
            program: vec![div(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 999), (11, 1)],
            expected_registers: vec![(12, 999)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_equal_numbers() {
        setup_tracing_with_log_level(Level::WARN);

        // 42 / 42 = 1
        let spec = TestSpec {
            program: vec![div(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 42), (11, 42)],
            expected_registers: vec![(12, 1)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_with_remainder() {
        setup_tracing_with_log_level(Level::WARN);

        // 17 / 5 = 3 (integer division, truncated)
        let spec = TestSpec {
            program: vec![div(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 17), (11, 5)],
            expected_registers: vec![(12, 3)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_zero_dividend() {
        setup_tracing_with_log_level(Level::WARN);

        // 0 / 100 = 0
        let spec = TestSpec {
            program: vec![div(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 0), (11, 100)],
            expected_registers: vec![(12, 0)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_large_numbers() {
        setup_tracing_with_log_level(Level::WARN);

        // 65536000 / 256 = 256000
        let spec = TestSpec {
            program: vec![divu(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 65536000), (11, 256)],
            expected_registers: vec![(12, 256000)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_powers_of_two() {
        setup_tracing_with_log_level(Level::WARN);

        // 128 / 8 = 16
        let spec = TestSpec {
            program: vec![div(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, 128), (11, 8)],
            expected_registers: vec![(12, 16)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_both_negative() {
        setup_tracing_with_log_level(Level::WARN);

        // (-20) / (-5) = 4
        let spec = TestSpec {
            program: vec![div(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, (-20_i32) as u32), (11, (-5_i32) as u32)],
            expected_registers: vec![(12, 4)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_chain() {
        setup_tracing_with_log_level(Level::WARN);

        // 120 / 2 = 60, 60 / 3 = 20
        let spec = TestSpec {
            program: vec![
                div(3, 0, 1), // reg[3] = 120 / 2 = 60
                div(4, 3, 2), // reg[4] = 60 / 3 = 20
            ],
            start_fp: 10,
            start_registers: vec![(10, 120), (11, 2), (12, 3)],
            expected_registers: vec![(14, 20)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_and_mul_inverse() {
        setup_tracing_with_log_level(Level::WARN);

        // (100 / 7) * 7 = 14 * 7 = 98 (not 100 due to integer truncation)
        let spec = TestSpec {
            program: vec![
                div(2, 0, 1), // reg[2] = 100 / 7 = 14
                mul(3, 2, 1), // reg[3] = 14 * 7 = 98
            ],
            start_fp: 10,
            start_registers: vec![(10, 100), (11, 7)],
            expected_registers: vec![(13, 98)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    // ==================== DivRem 64-bit Tests ====================

    #[test]
    fn test_div_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0000_0000_002A / 0x0000_0000_0000_0007 = 0x0000_0000_0000_0006
        let spec = TestSpec {
            program: vec![div_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 42),
                (125, 0), // reg 0 = 42
                (126, 7),
                (127, 0), // reg 2 = 7
            ],
            expected_registers: vec![(128, 6), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_64_large() {
        setup_tracing_with_log_level(Level::WARN);

        // 0x0000_0001_0000_0000 / 0x0000_0000_0000_0002 = 0x0000_0000_8000_0000
        let spec = TestSpec {
            program: vec![div_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 1), // reg 0 = 0x1_0000_0000
                (126, 2),
                (127, 0), // reg 2 = 2
            ],
            expected_registers: vec![(128, 0x8000_0000), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_divu_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 0xFFFF_FFFF_FFFF_FFFF / 0x0000_0000_0000_0002 = 0x7FFF_FFFF_FFFF_FFFF
        let spec = TestSpec {
            program: vec![divu_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0xFFFF_FFFF),
                (125, 0xFFFF_FFFF), // reg 0 = u64::MAX
                (126, 2),
                (127, 0), // reg 2 = 2
            ],
            expected_registers: vec![(128, 0xFFFF_FFFF), (129, 0x7FFF_FFFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_64_by_zero() {
        setup_tracing_with_log_level(Level::WARN);

        // 64-bit signed divide by zero returns -1 (all ones)
        let spec = TestSpec {
            program: vec![div_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 42),
                (125, 0), // reg 0 = 42
                (126, 0),
                (127, 0), // reg 2 = 0
            ],
            expected_registers: vec![(128, 0xFFFF_FFFF), (129, 0xFFFF_FFFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_divu_64_by_zero() {
        setup_tracing_with_log_level(Level::WARN);

        // 64-bit unsigned divide by zero returns all ones
        let spec = TestSpec {
            program: vec![divu_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 42),
                (125, 0), // reg 0 = 42
                (126, 0),
                (127, 0), // reg 2 = 0
            ],
            expected_registers: vec![(128, 0xFFFF_FFFF), (129, 0xFFFF_FFFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_remu_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 4294967297 % 3 = 2
        let spec = TestSpec {
            program: vec![remu_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 1),
                (125, 1), // reg 0 = 0x1_0000_0001 = 4294967297
                (126, 3),
                (127, 0), // reg 2 = 3
            ],
            expected_registers: vec![(128, 2), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_remu_64_by_zero() {
        setup_tracing_with_log_level(Level::WARN);

        // 64-bit unsigned remainder by zero returns dividend
        let spec = TestSpec {
            program: vec![remu_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 1),
                (125, 1), // reg 0 = 0x1_0000_0001
                (126, 0),
                (127, 0), // reg 2 = 0
            ],
            expected_registers: vec![(128, 1), (129, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_rems_64() {
        setup_tracing_with_log_level(Level::WARN);

        // -43 % 7 = -1 (64-bit signed)
        // -43 as i64 = 0xFFFF_FFFF_FFFF_FFD5
        // -1 as i64 = 0xFFFF_FFFF_FFFF_FFFF
        let spec = TestSpec {
            program: vec![rems_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0xFFFF_FFD5),
                (125, 0xFFFF_FFFF), // reg 0 = -43 as i64
                (126, 7),
                (127, 0), // reg 2 = 7
            ],
            expected_registers: vec![(128, 0xFFFF_FFFF), (129, 0xFFFF_FFFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_rems_64_by_zero() {
        setup_tracing_with_log_level(Level::WARN);

        // 64-bit signed remainder by zero returns dividend
        let spec = TestSpec {
            program: vec![rems_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0xFFFF_FFD5),
                (125, 0xFFFF_FFFF), // reg 0 = -43 as i64
                (126, 0),
                (127, 0), // reg 2 = 0
            ],
            expected_registers: vec![(128, 0xFFFF_FFD5), (129, 0xFFFF_FFFF)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_signed_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // i32::MIN / -1 = i32::MIN (RISC-V signed overflow returns dividend)
        let spec = TestSpec {
            program: vec![div(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, i32::MIN as u32), (11, (-1_i32) as u32)],
            expected_registers: vec![(12, i32::MIN as u32)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_rem_signed_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // i32::MIN % -1 = 0 (RISC-V signed overflow returns zero)
        let spec = TestSpec {
            program: vec![rems(2, 0, 1)],
            start_fp: 10,
            start_registers: vec![(10, i32::MIN as u32), (11, (-1_i32) as u32)],
            expected_registers: vec![(12, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_64_large_carry() {
        setup_tracing_with_log_level(Level::WARN);

        // 1 / (-1) = -1 (signed 64-bit)
        // This produces carries up to 4079 in the range tuple checker,
        // requiring sizes[1] >= 4096 (the 64-bit default).
        let spec = TestSpec {
            program: vec![div_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 1),
                (125, 0), // reg 0 = 1
                (126, 0xFFFFFFFF),
                (127, 0xFFFFFFFF), // reg 2 = -1 (i64)
            ],
            expected_registers: vec![(128, 0xFFFFFFFF), (129, 0xFFFFFFFF)], // -1
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_64_signed_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // i64::MIN / -1 = i64::MIN (RISC-V signed overflow returns dividend)
        let spec = TestSpec {
            program: vec![div_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 0x8000_0000), // reg 0 = i64::MIN
                (126, 0xFFFF_FFFF),
                (127, 0xFFFF_FFFF), // reg 2 = -1
            ],
            expected_registers: vec![(128, 0), (129, 0x8000_0000)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_rem_64_signed_overflow() {
        setup_tracing_with_log_level(Level::WARN);

        // i64::MIN % -1 = 0 (RISC-V signed overflow returns zero)
        let spec = TestSpec {
            program: vec![rems_64(4, 0, 2)],
            start_fp: 124,
            start_registers: vec![
                (124, 0),
                (125, 0x8000_0000), // reg 0 = i64::MIN
                (126, 0xFFFF_FFFF),
                (127, 0xFFFF_FFFF), // reg 2 = -1
            ],
            expected_registers: vec![(128, 0), (129, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== DivRem Immediate Tests ====================

    #[test]
    fn test_div_imm() {
        setup_tracing_with_log_level(Level::WARN);

        // 42 / 7 = 6
        let spec = TestSpec {
            program: vec![div_imm(1, 0, 7_i16)],
            start_fp: 10,
            start_registers: vec![(10, 42)],
            expected_registers: vec![(11, 6)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_divu_imm() {
        setup_tracing_with_log_level(Level::WARN);

        // 100 / 7 = 14
        let spec = TestSpec {
            program: vec![divu_imm(1, 0, 7_i16)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            expected_registers: vec![(11, 14)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_rems_imm() {
        setup_tracing_with_log_level(Level::WARN);

        // 43 % 7 = 1
        let spec = TestSpec {
            program: vec![rems_imm(1, 0, 7_i16)],
            start_fp: 10,
            start_registers: vec![(10, 43)],
            expected_registers: vec![(11, 1)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_remu_imm() {
        setup_tracing_with_log_level(Level::WARN);

        // 100 % 7 = 2
        let spec = TestSpec {
            program: vec![remu_imm(1, 0, 7_i16)],
            start_fp: 10,
            start_registers: vec![(10, 100)],
            expected_registers: vec![(11, 2)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_div_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 42 / 7 = 6 (64-bit)
        let spec = TestSpec {
            program: vec![div_imm_64(2, 0, 7_i16)],
            start_fp: 124,
            start_registers: vec![(124, 42), (125, 0)],
            expected_registers: vec![(126, 6), (127, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_divu_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 100 / 7 = 14 (64-bit)
        let spec = TestSpec {
            program: vec![divu_imm_64(2, 0, 7_i16)],
            start_fp: 124,
            start_registers: vec![(124, 100), (125, 0)],
            expected_registers: vec![(126, 14), (127, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_rems_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 43 % 7 = 1 (64-bit)
        let spec = TestSpec {
            program: vec![rems_imm_64(2, 0, 7_i16)],
            start_fp: 124,
            start_registers: vec![(124, 43), (125, 0)],
            expected_registers: vec![(126, 1), (127, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_remu_imm_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 100 % 7 = 2 (64-bit)
        let spec = TestSpec {
            program: vec![remu_imm_64(2, 0, 7_i16)],
            start_fp: 124,
            start_registers: vec![(124, 100), (125, 0)],
            expected_registers: vec![(126, 2), (127, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== Cross-width tests ====================

    #[test]
    fn test_cross_width_32_to_64() {
        setup_tracing_with_log_level(Level::WARN);

        // 32-bit writes 0x42 to reg fp+0, then 64-bit reads reg pair fp+0:fp+1
        let spec = TestSpec {
            program: vec![add_imm(0, 0, 0x42_i16), add_imm_64(2, 0, 0_i16)],
            start_fp: 10,
            expected_registers: vec![(10, 0x42), (12, 0x42), (13, 0)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_cross_width_64_to_32() {
        setup_tracing_with_log_level(Level::WARN);

        // 64-bit writes 0x42 to reg pair fp+0:fp+1, then 32-bit reads reg fp+0
        let spec = TestSpec {
            program: vec![add_imm_64(0, 0, 0x42_i16), add_imm(2, 0, 0_i16)],
            start_fp: 10,
            expected_registers: vec![(10, 0x42), (11, 0), (12, 0x42)],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    // ==================== I64Load / I64Load32U anti-dependency tests ====================
    //
    // These test the instruction orderings that arise during WASM-to-OpenVM
    // translation of 64-bit loads. The bugs were caused by register
    // anti-dependencies (WAR hazards) when the output register pair
    // overlaps the base address register.

    #[test]
    fn test_i64load_output_plus_1_eq_base_addr() {
        setup_tracing_with_log_level(Level::WARN);

        // Simulate I64Load where output+1 == base_addr:
        //   reg[fp+0]:reg[fp+1] = MEM64[reg[fp+1]]
        //
        // base_addr is reg fp+1, output lo is fp+0, output hi is fp+1.
        // The hi-word load must come AFTER the lo-word load, because the
        // hi-word destination (fp+1) clobbers the base address.
        //
        // Correct order: loadw(0, 1, 0), loadw(1, 1, 4)
        // Buggy order:   loadw(1, 1, 4), loadw(0, 1, 0)  — clobbers base before lo read
        let spec = TestSpec {
            program: vec![loadw(0, 1, 0), loadw(1, 1, 4)],
            start_fp: 10,
            start_registers: vec![(11, 200)], // reg fp+1 = address 200
            start_ram: vec![(200, 0xAAAA_BBBB), (204, 0xCCCC_DDDD)],
            expected_registers: vec![(10, 0xAAAA_BBBB), (11, 0xCCCC_DDDD)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_i64load_output_eq_base_addr() {
        setup_tracing_with_log_level(Level::WARN);

        // Simulate I64Load where output == base_addr:
        //   reg[fp+0]:reg[fp+1] = MEM64[reg[fp+0]]
        //
        // Here the lo-word load clobbers the base address, so the hi-word
        // load must come first.
        //
        // Correct order: loadw(1, 0, 4), loadw(0, 0, 0)
        let spec = TestSpec {
            program: vec![loadw(1, 0, 4), loadw(0, 0, 0)],
            start_fp: 10,
            start_registers: vec![(10, 200)], // reg fp+0 = address 200
            start_ram: vec![(200, 0x1111_2222), (204, 0x3333_4444)],
            expected_registers: vec![(10, 0x1111_2222), (11, 0x3333_4444)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_i64load32u_output_plus_1_eq_base_addr() {
        setup_tracing_with_log_level(Level::WARN);

        // Simulate I64Load32U where output+1 == base_addr:
        //   reg[fp+0] = MEM32[reg[fp+1]], reg[fp+1] = 0
        //
        // The CONST32 that zeros the hi word (fp+1) must come AFTER the
        // LOADW, because fp+1 is the base address register.
        //
        // Correct order: loadw(0, 1, 0), const_32_imm(1, 0, 0)
        // Buggy order:   const_32_imm(1, 0, 0), loadw(0, 1, 0)  — zeros base first
        let spec = TestSpec {
            program: vec![loadw(0, 1, 0), const_32_imm(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(11, 200)], // reg fp+1 = address 200
            start_ram: vec![(200, 0xDEAD_BEEF)],
            expected_registers: vec![(10, 0xDEAD_BEEF), (11, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_i64load32u_no_overlap() {
        setup_tracing_with_log_level(Level::WARN);

        // Simulate I64Load32U with no overlap (output != base_addr, output+1 != base_addr):
        //   reg[fp+0] = MEM32[reg[fp+2]], reg[fp+1] = 0
        //
        // Order doesn't matter here, but verify the result is correct.
        let spec = TestSpec {
            program: vec![loadw(0, 2, 0), const_32_imm(1, 0, 0)],
            start_fp: 10,
            start_registers: vec![(12, 200)], // reg fp+2 = address 200
            start_ram: vec![(200, 0xCAFE_BABE)],
            expected_registers: vec![(10, 0xCAFE_BABE), (11, 0)],
            ..Default::default()
        };

        test_spec_for_all_register_bases(spec)
    }

    // ==================== Call Tests ====================
    //
    // Call instructions (RET, CALL, CALL_INDIRECT) change the frame pointer (FP)
    // and jump to a new PC.
    // Register accesses are FP-relative: register N at FP=F is at absolute
    // register index (F + N), raw address ((F + N) * 4).
    //
    // Notation:
    //   fp(L)  = logical FP L, raw FP = L * 4
    //   reg[N] = FP-relative register N, absolute index = logical_fp + N
    //
    // FP semantics:
    //   CALL/CALL_INDIRECT: new_fp = current_fp + offset_from_register
    //   RET: new_fp = absolute_fp_from_register
    //
    // Common setup: caller at fp(20), callee at fp(50) (offset = 30).

    #[test]
    fn test_ret() {
        setup_tracing_with_log_level(Level::WARN);

        // RET: Restore PC and FP from registers, then operate in the restored frame.
        //
        // Start at fp(50) (raw 200, callee frame).
        // reg[10] at abs 60 = 8 (return PC), reg[11] at abs 61 = 80 (caller raw FP → fp(20)).
        // Pre-populate caller_frame[5] at abs 25 = 99.
        //
        // PC=0: RET → PC=8, FP=80 → fp(20)
        // PC=4: skipped
        // PC=8: add_imm caller_frame[0] = caller_frame[5] + 1 = 99 + 1 = 100
        //
        // Verifies: FP restored, post-return instruction operates in the caller frame.
        let spec = TestSpec {
            program: vec![
                ret(10, 11), // PC=0: return to PC=8, FP=reg[11]
                halt(),      // PC=4: skipped
                add_imm(0, 5, 1_i16), // PC=8: caller[0] = caller[5] + 1
                             // PC=12: halt (appended by test_spec)
            ],
            start_fp: 50,
            start_registers: vec![
                (60, 8),  // reg[10] at fp(50): return PC
                (61, 80), // reg[11] at fp(50): caller raw FP
                (25, 99), // caller_frame[5] at abs 20+5=25
            ],
            expected_pc: Some(12),
            expected_fp: Some(20),
            expected_registers: vec![
                (20, 100), // caller_frame[0] at abs 20: 99 + 1
                (25, 99),  // caller_frame[5] unchanged
            ],
            fp_value_registers: vec![61],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_call() {
        setup_tracing_with_log_level(Level::WARN);

        // CALL: Save PC and FP, jump to immediate PC with new FP, then compute in new frame.
        //
        // Start at fp(20) (raw 80). FP offset = 120 (immediate): 80 + 120 = 200 → fp(50).
        // Pre-populate new_frame[3] at abs 53 = 55.
        // call(save_pc=10, save_fp=11, to_pc=12, fp_offset=120)
        //
        // Saves: return PC=4 → new_frame[10] (abs 60), old FP=80 → new_frame[11] (abs 61).
        // PC=12: add_imm new_frame[0] = new_frame[3] + 7 = 55 + 7 = 62
        //
        // Verifies: saved PC/FP and post-call computation in the new frame.
        let spec = TestSpec {
            program: vec![
                call(10, 11, 12, 120), // PC=0: call to PC=12, FP offset=120
                halt(),                // PC=4: skipped (return would land here)
                halt(),                // PC=8: skipped
                add_imm(0, 3, 7_i16),  // PC=12: new_frame[0] = new_frame[3]+7
                                       // PC=16: halt (appended by test_spec)
            ],
            start_fp: 20,
            start_registers: vec![
                (53, 55), // new_frame[3] at abs 53: pre-populated
            ],
            expected_pc: Some(16),
            expected_fp: Some(50),
            expected_registers: vec![
                (50, 62), // new_frame[0] at abs 50: 55 + 7
                (60, 4),  // new_frame[10] at abs 60: saved return PC
                (61, 80), // new_frame[11] at abs 61: saved old raw FP
            ],
            fp_value_registers: vec![61],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_call_indirect() {
        setup_tracing_with_log_level(Level::WARN);

        // CALL_INDIRECT: Save PC and FP, jump to register PC with new FP, then compute.
        //
        // Start at fp(20) (raw 80). FP offset = 120 (immediate): 80 + 120 = 200 → fp(50).
        // reg[12] at abs 32 = 12 (target PC).
        // Pre-populate new_frame[3] at abs 53 = 55.
        // call_indirect(save_pc=10, save_fp=11, to_pc_reg=12, fp_offset=120)
        //
        // Saves: return PC=4 → abs 60, old FP=80 → abs 61.
        // PC=12: add_imm new_frame[0] = new_frame[3] + 7 = 55 + 7 = 62
        //
        // Verifies: saved PC/FP and post-call computation, with PC from register.
        let spec = TestSpec {
            program: vec![
                call_indirect(10, 11, 12, 120), // PC=0: call indirect, FP offset=120
                halt(),                         // PC=4: skipped
                halt(),                         // PC=8: skipped
                add_imm(0, 3, 7_i16),           // PC=12: new_frame[0] = new_frame[3]+7
                                                // PC=16: halt (appended by test_spec)
            ],
            start_fp: 20,
            start_registers: vec![
                (32, 12), // reg[12] at fp(20): target PC
                (53, 55), // new_frame[3] at abs 53: pre-populated
            ],
            expected_pc: Some(16),
            expected_fp: Some(50),
            expected_registers: vec![
                (50, 62), // new_frame[0] at abs 50: 55 + 7
                (60, 4),  // new_frame[10] at abs 60: saved return PC
                (61, 80), // new_frame[11] at abs 61: saved old raw FP
            ],
            fp_value_registers: vec![61],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_call_and_return() {
        setup_tracing_with_log_level(Level::WARN);

        // Complete call + return sequence with computation in both frames.
        //
        // Start at fp(20) (raw 80). caller_frame[5] (abs 25) = 100.
        // FP offset = 120 (immediate): 80 + 120 = 200 → fp(50).
        //
        // PC=0:  CALL(save_pc=10, save_fp=11, to_pc=16, fp_offset=120)
        //        saves return PC=4 → abs 60, old FP=80 → abs 61
        //        jumps to PC=16, FP=200
        // PC=4:  (return lands here) add_imm caller[0] = caller[5] + 1 = 101
        // PC=8:  halt (after return, verifies caller frame computation)
        // PC=12: skipped padding
        // PC=16: add_imm callee[3] = callee[3] + 42 = 0+42 = 42
        // PC=20: RET(10, 11) → PC=4, FP=80
        //
        // After return: caller does add_imm, then halts.
        // Verifies: round-trip call/return, computation in both frames persists.
        let spec = TestSpec {
            program: vec![
                call(10, 11, 16, 120), // PC=0: call to PC=16, FP offset=120
                add_imm(0, 5, 1_i16),  // PC=4: caller[0] = caller[5]+1 (after return)
                halt(),                // PC=8: halt after return
                halt(),                // PC=12: padding
                add_imm(3, 3, 42_i16), // PC=16: callee[3] = 0 + 42
                ret(10, 11),           // PC=20: return to caller
            ],
            start_fp: 20,
            start_registers: vec![
                (25, 100), // caller_frame[5] at abs 25
            ],
            expected_pc: Some(8),
            expected_fp: Some(20), // returned to caller
            expected_registers: vec![
                (20, 101), // caller_frame[0] at abs 20: 100 + 1
                (25, 100), // caller_frame[5] unchanged
                (53, 42),  // callee_frame[3] at abs 53: written by callee, persists
            ],
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    // ==================== Hint/Stdin Tests ====================

    #[test]
    fn test_input_hint() {
        setup_tracing_with_log_level(Level::WARN);

        let mut stdin = StdIn::default();
        stdin.write(&42u32);

        // Register 0 defaults to 0 (memory pointer to MEM[0]).
        // Read a u32 from stdin via hint mechanism and verify RAM value.
        let spec = TestSpec {
            program: vec![
                prepare_read(),
                hint_storew(0), // skip length word
                hint_storew(0), // write data to MEM[0]
            ],
            expected_ram: vec![(0, 42)],
            stdin,
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    #[test]
    fn test_input_hint_with_frame_jump_and_xor() {
        setup_tracing_with_log_level(Level::WARN);

        let mut stdin = StdIn::default();
        stdin.write(&170u32);
        stdin.write(&204u32);

        // Read two values from stdin across a frame change (CALL) and XOR them.
        // Frame 1 (FP=0): read first value, store to MEM[100], call
        // Frame 2 (FP=50): load from MEM[100], read second value, XOR
        let scratch = 5;
        let spec = TestSpec {
            program: vec![
                const_32_imm(0, 0, 0),       // PC=0
                const_32_imm(scratch, 0, 0), // PC=4: scratch = 0
                prepare_read(),              // PC=8
                hint_storew(scratch),        // PC=12: skip length
                hint_storew(scratch),        // PC=16: write data to MEM[0]
                loadw(8, scratch, 0),        // PC=20: r8 = MEM[0]
                add_imm(6, 0, 100_i16),      // PC=24: r6 = 100
                storew(8, 6, 0),             // PC=28: MEM[100] = r8
                call(10, 11, 40, 200),       // PC=32: call to PC=40, FP += 200
                halt(),                      // PC=36: padding
                // === New frame (PC=40), raw FP = start_fp*4+200, logical FP = start_fp+50 ===
                const_32_imm(0, 0, 0),       // PC=40
                const_32_imm(scratch, 0, 0), // PC=44: scratch = 0
                add_imm(6, 0, 100_i16),      // PC=48: r6 = 100
                loadw(2, 6, 0),              // PC=52: r2 = MEM[100]
                prepare_read(),              // PC=56
                hint_storew(scratch),        // PC=60: skip length
                hint_storew(scratch),        // PC=64: write data to MEM[0]
                loadw(3, scratch, 0),        // PC=68: r3 = MEM[0]
                xor(4, 2, 3),                // PC=72: r4 = r2 ^ r3
            ],
            expected_fp: Some(50),
            expected_registers: vec![(54, 170 ^ 204)],
            expected_ram: vec![(0, 204), (100, 170)],
            stdin,
            ..Default::default()
        };
        test_spec_for_all_register_bases(spec)
    }

    // ==================== Non-TestSpec Tests ====================
    // These tests require special infrastructure (error handling)
    // that doesn't fit the TestSpec framework.

    #[test]
    fn test_trap() {
        use crush_translation::ERROR_CODE_OFFSET;

        setup_tracing_with_log_level(Level::WARN);

        let instructions: Vec<Instruction<F>> = vec![trap(42), halt()];
        let program = Program::from_instructions(&instructions);
        let exe = VmExe::new(program);
        let vm_config = CrushConfig::default();
        let vm = VmExecutor::new(vm_config).unwrap();
        let instance = vm.instance(&exe).unwrap();
        match instance.execute(StdIn::default(), None) {
            Err(ExecutionError::FailedWithExitCode(code)) => {
                assert_eq!(code, ERROR_CODE_OFFSET + 42);
            }
            Err(other) => panic!("Expected FailedWithExitCode, got: {other:?}"),
            Ok(_) => panic!("Expected execution error, but succeeded"),
        }
    }
}
