use std::collections::{BTreeSet, HashSet};

use crush_circuit::{CrushConfig, CrushConfigExecutor, CrushCpuBuilder, CrushCpuProverExt};
use crush_translation::LinkedProgram;
use openvm_circuit::arch::{
    AirInventory, AirInventoryError, ChipInventoryError, VmBuilder, VmCircuitConfig,
    VmCircuitExtension, VmProverExtension,
};
use openvm_circuit::system::SystemCpuBuilder;
#[cfg(feature = "cuda")]
use openvm_circuit::system::cuda::extensions::SystemGpuBuilder;
use openvm_crush_transpiler::{
    BaseAlu64Opcode, BaseAluOpcode, CallOpcode, ConstOpcodes, DivRem64Opcode, DivRemOpcode,
    Eq64Opcode, EqOpcode, JumpOpcode, LessThan64Opcode, LessThanOpcode, LoadStoreOpcode,
    Mul64Opcode, MulOpcode, Shift64Opcode, ShiftOpcode,
};
use openvm_instructions::{
    LocalOpcode, VmOpcode,
    instruction::Instruction,
    riscv::{RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine, p3_baby_bear::BabyBear,
};
use powdr_autoprecompiles::DropHint;
use powdr_openvm::BabyBearSC;
#[cfg(feature = "cuda")]
use powdr_openvm::GpuBabyBearPoseidon2CpuEngine;
use powdr_openvm::drop_hint_lowering::DropHintConfig;
#[cfg(feature = "cuda")]
use powdr_openvm::isa::OriginalGpuChipComplex;
use powdr_openvm::isa::{OpenVmISA, OriginalCpuChipComplex};
use powdr_openvm::powdr_extension::trace_generator::SharedPeripheryChipsCpu;
#[cfg(feature = "cuda")]
use powdr_openvm::powdr_extension::trace_generator::SharedPeripheryChipsGpu;
use powdr_openvm::powdr_extension::trace_generator::cpu::SharedPeripheryChipsCpuProverExt;
use powdr_openvm::program::OriginalCompiledProgram;
use powdr_riscv_elf::debug_info::SymbolTable;
use strum::IntoEnumIterator;

mod formatter;
use formatter::crush_instruction_formatter;

#[derive(Clone, Default)]
pub struct CrushISA;

impl OpenVmISA for CrushISA {
    type LinkedProgram<'a> = LinkedProgram<BabyBear>;
    type Executor<F: openvm_circuit::arch::VmField> = CrushConfigExecutor<F>;
    type Config = CrushConfig;
    type CpuBuilder = CrushCpuBuilder;
    #[cfg(feature = "cuda")]
    type GpuBuilder = crush_circuit::CrushGpuBuilder;

    fn create_original_chip_complex(
        config: &Self::Config,
        airs: AirInventory<BabyBearSC>,
    ) -> Result<OriginalCpuChipComplex, ChipInventoryError> {
        <CrushCpuBuilder as VmBuilder<BabyBearPoseidon2CpuEngine>>::create_chip_complex(
            &CrushCpuBuilder,
            config,
            airs,
        )
    }

    fn branching_opcodes() -> HashSet<VmOpcode> {
        let mut set = HashSet::new();
        set.extend(JumpOpcode::iter().map(|x| x.global_opcode()));
        set.extend(CallOpcode::iter().map(|x| x.global_opcode()));
        set
    }

    fn allowed_opcodes() -> HashSet<VmOpcode> {
        let mut set = HashSet::new();
        set.extend(BaseAluOpcode::iter().map(|x| x.global_opcode()));
        set.extend(BaseAlu64Opcode::iter().map(|x| x.global_opcode()));
        set.extend(MulOpcode::iter().map(|x| x.global_opcode()));
        set.extend(Mul64Opcode::iter().map(|x| x.global_opcode()));
        set.extend(LessThanOpcode::iter().map(|x| x.global_opcode()));
        set.extend(LessThan64Opcode::iter().map(|x| x.global_opcode()));
        set.extend(DivRemOpcode::iter().map(|x| x.global_opcode()));
        set.extend(DivRem64Opcode::iter().map(|x| x.global_opcode()));
        set.extend(EqOpcode::iter().map(|x| x.global_opcode()));
        set.extend(Eq64Opcode::iter().map(|x| x.global_opcode()));
        set.extend(ShiftOpcode::iter().map(|x| x.global_opcode()));
        set.extend(Shift64Opcode::iter().map(|x| x.global_opcode()));
        set.extend(
            LoadStoreOpcode::iter()
                .take(LoadStoreOpcode::STOREB as usize + 1)
                .map(|x| x.global_opcode()),
        );
        set.extend([LoadStoreOpcode::LOADB, LoadStoreOpcode::LOADH].map(|x| x.global_opcode()));
        set.extend(JumpOpcode::iter().map(|x| x.global_opcode()));
        set.extend(CallOpcode::iter().map(|x| x.global_opcode()));
        set.extend(ConstOpcodes::iter().map(|x| x.global_opcode()));
        set
    }

    fn format<F: PrimeField32>(instruction: &Instruction<F>) -> String {
        crush_instruction_formatter(instruction)
    }

    fn get_symbol_table<'a>(program: &Self::LinkedProgram<'a>) -> SymbolTable {
        SymbolTable::from_table(
            program
                .labels()
                .into_iter()
                .map(|(key, values)| (key.try_into().unwrap(), values))
                .collect(),
        )
    }

    fn get_jump_destinations(original_program: &OriginalCompiledProgram<Self>) -> BTreeSet<u64> {
        original_program
            .linked_program
            .labels()
            .into_keys()
            .collect()
    }

    fn get_drop_hints<'a>(
        original_program: &'a OriginalCompiledProgram<'_, Self>,
    ) -> &'a [Vec<DropHint>] {
        // The linked program's hints are indexed by word-PC and include a
        // leading empty slot for the linker's reserved nop at PC 0, which
        // `program_with_entry_point` strips from the exe (its `pc_base` is one
        // pc step). Skipping that slot aligns entry `i` with the instruction
        // at `pc_base + i * DEFAULT_PC_STEP`, as required by this hook.
        &original_program.linked_program.drop_hints()[1..]
    }

    fn drop_hint_config() -> Option<DropHintConfig> {
        Some(DropHintConfig {
            // Hint register indices are FP-relative words; the instruction
            // builders encode register operands as `index * NUM_LIMBS`, so a
            // hint's slot address is `fp + index * NUM_LIMBS`.
            register_address_space: RV32_REGISTER_AS,
            stride: RV32_REGISTER_NUM_LIMBS as u64,
            fp_address_space: crush_circuit::memory_config::FP_AS,
            fp_address: 0,
        })
    }

    fn read_fp_untraced(memory: &openvm_circuit::system::memory::online::GuestMemory) -> u32 {
        use crush_circuit::memory_config::FpMemory;
        memory.fp::<BabyBear>()
    }

    fn create_dummy_airs<E: VmCircuitExtension<BabyBearSC>>(
        config: &Self::Config,
        shared_chips: E,
    ) -> Result<AirInventory<BabyBearSC>, AirInventoryError> {
        let mut inventory = config.system.create_airs()?;
        inventory.start_new_extension();
        VmCircuitExtension::extend_circuit(&shared_chips, &mut inventory)?;
        VmCircuitExtension::extend_circuit(&config.base, &mut inventory)?;
        Ok(inventory)
    }

    fn create_dummy_chip_complex_cpu(
        config: &Self::Config,
        circuit: AirInventory<BabyBearSC>,
        shared_chips: SharedPeripheryChipsCpu<Self>,
    ) -> Result<OriginalCpuChipComplex, ChipInventoryError> {
        let mut chip_complex = VmBuilder::<BabyBearPoseidon2CpuEngine>::create_chip_complex(
            &SystemCpuBuilder,
            &config.system,
            circuit,
        )?;
        let inventory = &mut chip_complex.inventory;

        VmProverExtension::<BabyBearPoseidon2CpuEngine, _, _>::extend_prover(
            &SharedPeripheryChipsCpuProverExt,
            &shared_chips,
            inventory,
        )?;

        VmProverExtension::<BabyBearPoseidon2CpuEngine, _, _>::extend_prover(
            &CrushCpuProverExt,
            &config.base,
            inventory,
        )?;

        Ok(chip_complex)
    }

    #[cfg(feature = "cuda")]
    fn create_dummy_chip_complex_gpu(
        config: &Self::Config,
        circuit: AirInventory<BabyBearSC>,
        shared_chips: SharedPeripheryChipsGpu<Self>,
    ) -> Result<OriginalGpuChipComplex, ChipInventoryError> {
        use powdr_openvm::powdr_extension::trace_generator::cuda::SharedPeripheryChipsGpuProverExt;

        let mut chip_complex = VmBuilder::<GpuBabyBearPoseidon2CpuEngine>::create_chip_complex(
            &SystemGpuBuilder,
            &config.system,
            circuit,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::extend_prover(
            &SharedPeripheryChipsGpuProverExt,
            &shared_chips,
            inventory,
        )?;
        VmProverExtension::extend_prover(
            &crush_circuit::CrushGpuProverExt,
            &config.base,
            inventory,
        )?;
        Ok(chip_complex)
    }
}

#[cfg(test)]
mod tests {
    use powdr_openvm::extraction_utils::OriginalVmConfig;

    use super::*;

    #[test]
    fn machine_extraction() {
        let generate = powdr_openvm::default_generate_config();
        let original_config = OriginalVmConfig::<CrushISA>::new(CrushConfig::default());
        let _ = original_config.airs(generate.degree_bound).unwrap();
    }
}
