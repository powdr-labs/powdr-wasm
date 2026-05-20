use std::collections::{BTreeSet, HashSet};

use crush_circuit::{CrushConfig, CrushConfigExecutor, CrushCpuBuilder, CrushCpuProverExt};
use crush_translation::LinkedProgram;
use keccak_circuit::Keccak256CpuProverExt;
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
use openvm_instructions::{LocalOpcode, VmOpcode, instruction::Instruction};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine, p3_baby_bear::BabyBear,
};
use powdr_openvm::BabyBearSC;
#[cfg(feature = "cuda")]
use powdr_openvm::GpuBabyBearPoseidon2Engine;
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
    type LinkedProgram<'a> = LinkedProgram<'a, BabyBear>;
    type Executor<F: PrimeField32> = CrushConfigExecutor<F>;
    type Config = CrushConfig;
    type CpuBuilder = CrushCpuBuilder;
    #[cfg(feature = "cuda")]
    type GpuBuilder = crush_circuit::CrushGpuBuilder;

    fn create_original_chip_complex(
        config: &Self::Config,
        airs: AirInventory<BabyBearSC>,
    ) -> Result<OriginalCpuChipComplex, ChipInventoryError> {
        <CrushCpuBuilder as VmBuilder<BabyBearPoseidon2Engine>>::create_chip_complex(
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

    fn create_dummy_airs<E: VmCircuitExtension<BabyBearSC>>(
        config: &Self::Config,
        shared_chips: E,
    ) -> Result<AirInventory<BabyBearSC>, AirInventoryError> {
        let mut inventory = config.system.create_airs()?;
        inventory.start_new_extension();
        VmCircuitExtension::extend_circuit(&shared_chips, &mut inventory)?;
        VmCircuitExtension::extend_circuit(&config.base, &mut inventory)?;
        if let Some(ref keccak) = config.keccak {
            VmCircuitExtension::extend_circuit(keccak, &mut inventory)?;
        }
        Ok(inventory)
    }

    fn create_dummy_chip_complex_cpu(
        config: &Self::Config,
        circuit: AirInventory<BabyBearSC>,
        shared_chips: SharedPeripheryChipsCpu<Self>,
    ) -> Result<OriginalCpuChipComplex, ChipInventoryError> {
        let mut chip_complex = VmBuilder::<BabyBearPoseidon2Engine>::create_chip_complex(
            &SystemCpuBuilder,
            &config.system,
            circuit,
        )?;
        let inventory = &mut chip_complex.inventory;

        VmProverExtension::<BabyBearPoseidon2Engine, _, _>::extend_prover(
            &SharedPeripheryChipsCpuProverExt,
            &shared_chips,
            inventory,
        )?;

        VmProverExtension::<BabyBearPoseidon2Engine, _, _>::extend_prover(
            &CrushCpuProverExt,
            &config.base,
            inventory,
        )?;

        if let Some(ref keccak) = config.keccak {
            VmProverExtension::<BabyBearPoseidon2Engine, _, _>::extend_prover(
                &Keccak256CpuProverExt,
                keccak,
                inventory,
            )?;
        }

        Ok(chip_complex)
    }

    #[cfg(feature = "cuda")]
    fn create_dummy_chip_complex_gpu(
        config: &Self::Config,
        circuit: AirInventory<BabyBearSC>,
        shared_chips: SharedPeripheryChipsGpu<Self>,
    ) -> Result<OriginalGpuChipComplex, ChipInventoryError> {
        use powdr_openvm::powdr_extension::trace_generator::cuda::SharedPeripheryChipsGpuProverExt;

        let mut chip_complex = VmBuilder::<GpuBabyBearPoseidon2Engine>::create_chip_complex(
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
        let powdr_config = powdr_openvm::default_powdr_openvm_config(0, 0);
        let original_config = OriginalVmConfig::<CrushISA>::new(CrushConfig::default());
        let _ = original_config.airs(powdr_config.degree_bound).unwrap();
    }
}
