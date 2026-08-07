use std::sync::Arc;

use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, DEFAULT_BLOCK_SIZE,
        ExecutionBridge, ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena,
        VmCircuitExtension, VmExecutionExtension, VmProverExtension,
    },
    system::{SystemPort, memory::SharedMemoryHelper},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    var_range::VariableRangeCheckerBus,
};
use openvm_cpu_backend::{CpuBackend, CpuDevice};
// Plain curve data, with no frame-pointer involvement: reuse upstream's rather than
// duplicating it, so the pairing extension's `CurveConfig` is the same type as ours.
pub use openvm_ecc_circuit::{
    CurveConfig, P256_CONFIG, P256_ECC_STRUCT_NAME, P256_MODULUS, P256_ORDER, SECP256K1_CONFIG,
    SECP256K1_ECC_STRUCT_NAME, SECP256K1_MODULUS, SECP256K1_ORDER,
};
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::{LocalOpcode, VmOpcode};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_stark_backend::{StarkEngine, StarkProtocolConfig, Val, p3_field::PrimeField32};
use serde::{Deserialize, Serialize};
use strum::EnumCount;

use crate::ecc::{
    ECC_BLOCKS_32, ECC_BLOCKS_48, EcAddNeExecutor, EcDoubleExecutor, EccCpuProverExt, NUM_LIMBS_32,
    NUM_LIMBS_48, WeierstrassAir, get_ec_addne_air, get_ec_addne_chip, get_ec_addne_step,
    get_ec_double_air, get_ec_double_chip, get_ec_double_step,
};

#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct WeierstrassExtension {
    pub supported_curves: Vec<CurveConfig>,
}

impl WeierstrassExtension {
    pub fn generate_sw_init(&self) -> String {
        let supported_curves = self
            .supported_curves
            .iter()
            .map(|curve_config| format!("\"{}\"", curve_config.struct_name))
            .collect::<Vec<String>>()
            .join(", ");

        format!("openvm_ecc_guest::sw_macros::sw_init! {{ {supported_curves} }}")
    }
}

#[derive(Clone, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
pub enum WeierstrassExtensionExecutor {
    // 32 limbs prime
    EcAddNeRv32_32(EcAddNeExecutor<ECC_BLOCKS_32, DEFAULT_BLOCK_SIZE>),
    EcDoubleRv32_32(EcDoubleExecutor<ECC_BLOCKS_32, DEFAULT_BLOCK_SIZE>),
    // 48 limbs prime
    EcAddNeRv32_48(EcAddNeExecutor<ECC_BLOCKS_48, DEFAULT_BLOCK_SIZE>),
    EcDoubleRv32_48(EcDoubleExecutor<ECC_BLOCKS_48, DEFAULT_BLOCK_SIZE>),
}

impl<F: PrimeField32> VmExecutionExtension<F> for WeierstrassExtension {
    type Executor = WeierstrassExtensionExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, WeierstrassExtensionExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();
        // TODO: somehow get the range checker bus from `ExecutorInventory`
        let dummy_range_checker_bus = VariableRangeCheckerBus::new(u16::MAX, 16);
        for (i, curve) in self.supported_curves.iter().enumerate() {
            let start_offset =
                Rv32WeierstrassOpcode::CLASS_OFFSET + i * Rv32WeierstrassOpcode::COUNT;
            let bytes = curve.modulus.bits().div_ceil(8) as usize;

            if bytes <= NUM_LIMBS_32 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: NUM_LIMBS_32,
                    limb_bits: 8,
                };
                let addne = get_ec_addne_step(
                    config.clone(),
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    WeierstrassExtensionExecutor::EcAddNeRv32_32(addne),
                    ((Rv32WeierstrassOpcode::EC_ADD_NE as usize)
                        ..=(Rv32WeierstrassOpcode::SETUP_EC_ADD_NE as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;

                let double = get_ec_double_step(
                    config,
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                    curve.a.clone(),
                );

                inventory.add_executor(
                    WeierstrassExtensionExecutor::EcDoubleRv32_32(double),
                    ((Rv32WeierstrassOpcode::EC_DOUBLE as usize)
                        ..=(Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
            } else if bytes <= NUM_LIMBS_48 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: NUM_LIMBS_48,
                    limb_bits: 8,
                };
                let addne = get_ec_addne_step(
                    config.clone(),
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                );

                inventory.add_executor(
                    WeierstrassExtensionExecutor::EcAddNeRv32_48(addne),
                    ((Rv32WeierstrassOpcode::EC_ADD_NE as usize)
                        ..=(Rv32WeierstrassOpcode::SETUP_EC_ADD_NE as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;

                let double = get_ec_double_step(
                    config,
                    dummy_range_checker_bus,
                    pointer_max_bits,
                    start_offset,
                    curve.a.clone(),
                );

                inventory.add_executor(
                    WeierstrassExtensionExecutor::EcDoubleRv32_48(double),
                    ((Rv32WeierstrassOpcode::EC_DOUBLE as usize)
                        ..=(Rv32WeierstrassOpcode::SETUP_EC_DOUBLE as usize))
                        .map(|x| VmOpcode::from_usize(x + start_offset)),
                )?;
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

impl<SC: StarkProtocolConfig> VmCircuitExtension<SC> for WeierstrassExtension {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let range_checker_bus = inventory.range_checker().bus;
        let pointer_max_bits = inventory.pointer_max_bits();

        let bitwise_lu = {
            // A trick to get around Rust's borrow rules
            let existing_air = inventory.find_air::<BitwiseOperationLookupAir<8>>().next();
            if let Some(air) = existing_air {
                air.bus
            } else {
                let bus = BitwiseOperationLookupBus::new(inventory.new_bus_idx());
                let air = BitwiseOperationLookupAir::<8>::new(bus);
                inventory.add_air(air);
                air.bus
            }
        };
        for (i, curve) in self.supported_curves.iter().enumerate() {
            let start_offset =
                Rv32WeierstrassOpcode::CLASS_OFFSET + i * Rv32WeierstrassOpcode::COUNT;
            let bytes = curve.modulus.bits().div_ceil(8) as usize;

            if bytes <= NUM_LIMBS_32 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: NUM_LIMBS_32,
                    limb_bits: 8,
                };

                let addne = get_ec_addne_air::<ECC_BLOCKS_32, DEFAULT_BLOCK_SIZE>(
                    exec_bridge,
                    memory_bridge,
                    config.clone(),
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                );
                inventory.add_air(addne);

                let double = get_ec_double_air::<ECC_BLOCKS_32, DEFAULT_BLOCK_SIZE>(
                    exec_bridge,
                    memory_bridge,
                    config,
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                    curve.a.clone(),
                );
                inventory.add_air(double);
            } else if bytes <= NUM_LIMBS_48 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: NUM_LIMBS_48,
                    limb_bits: 8,
                };

                let addne = get_ec_addne_air::<ECC_BLOCKS_48, DEFAULT_BLOCK_SIZE>(
                    exec_bridge,
                    memory_bridge,
                    config.clone(),
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                );
                inventory.add_air(addne);

                let double = get_ec_double_air::<ECC_BLOCKS_48, DEFAULT_BLOCK_SIZE>(
                    exec_bridge,
                    memory_bridge,
                    config,
                    range_checker_bus,
                    bitwise_lu,
                    pointer_max_bits,
                    start_offset,
                    curve.a.clone(),
                );
                inventory.add_air(double);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<SC, E, RA> VmProverExtension<E, RA, WeierstrassExtension> for EccCpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        extension: &WeierstrassExtension,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let bitwise_lu = {
            let existing_chip = inventory
                .find_chip::<SharedBitwiseOperationLookupChip<8>>()
                .next();
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let air: &BitwiseOperationLookupAir<8> = inventory.next_air()?;
                let chip = Arc::new(BitwiseOperationLookupChip::new(air.bus));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };
        for curve in extension.supported_curves.iter() {
            let bytes = curve.modulus.bits().div_ceil(8) as usize;

            if bytes <= NUM_LIMBS_32 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: NUM_LIMBS_32,
                    limb_bits: 8,
                };

                inventory.next_air::<WeierstrassAir<2, ECC_BLOCKS_32, DEFAULT_BLOCK_SIZE>>()?;
                let addne = get_ec_addne_chip::<Val<SC>, ECC_BLOCKS_32, DEFAULT_BLOCK_SIZE>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(addne);

                inventory.next_air::<WeierstrassAir<1, ECC_BLOCKS_32, DEFAULT_BLOCK_SIZE>>()?;
                let double = get_ec_double_chip::<Val<SC>, ECC_BLOCKS_32, DEFAULT_BLOCK_SIZE>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                    curve.a.clone(),
                );
                inventory.add_executor_chip(double);
            } else if bytes <= NUM_LIMBS_48 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: NUM_LIMBS_48,
                    limb_bits: 8,
                };

                inventory.next_air::<WeierstrassAir<2, ECC_BLOCKS_48, DEFAULT_BLOCK_SIZE>>()?;
                let addne = get_ec_addne_chip::<Val<SC>, ECC_BLOCKS_48, DEFAULT_BLOCK_SIZE>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(addne);

                inventory.next_air::<WeierstrassAir<1, ECC_BLOCKS_48, DEFAULT_BLOCK_SIZE>>()?;
                let double = get_ec_double_chip::<Val<SC>, ECC_BLOCKS_48, DEFAULT_BLOCK_SIZE>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                    curve.a.clone(),
                );
                inventory.add_executor_chip(double);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}
