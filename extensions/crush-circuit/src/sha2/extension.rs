use std::{
    result::Result,
    sync::{Arc, Mutex},
};

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError,
        ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena, VmCircuitExtension,
        VmExecutionExtension, VmField, VmProverExtension,
    },
    system::memory::SharedMemoryHelper,
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_cpu_backend::{CpuBackend, CpuDevice};
use openvm_crush_transpiler::Sha2Opcode;
use openvm_instructions::LocalOpcode;
use openvm_sha2_air::{Sha256Config, Sha512Config};
use openvm_stark_backend::{StarkEngine, StarkProtocolConfig, Val};
use serde::{Deserialize, Serialize};

use crate::sha2::{
    Sha2BlockHasherChip, Sha2BlockHasherVmAir, Sha2MainAir, Sha2MainChip, Sha2VmExecutor,
};

// =================================== VM Extension Implementation =================================

/// The SHA-2 extension: the `SHA256` and `SHA512` compression opcodes, each paired
/// with a block-hasher periphery chip over a shared bus.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Sha2;

#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
pub enum Sha2Executor {
    Sha256(Sha2VmExecutor<Sha256Config>),
    Sha512(Sha2VmExecutor<Sha512Config>),
}

impl<F> VmExecutionExtension<F> for Sha2 {
    type Executor = Sha2Executor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Sha2Executor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();

        let sha256_executor =
            Sha2VmExecutor::<Sha256Config>::new(Sha2Opcode::CLASS_OFFSET, pointer_max_bits);
        inventory.add_executor(sha256_executor, [Sha2Opcode::SHA256.global_opcode()])?;

        let sha512_executor =
            Sha2VmExecutor::<Sha512Config>::new(Sha2Opcode::CLASS_OFFSET, pointer_max_bits);
        inventory.add_executor(sha512_executor, [Sha2Opcode::SHA512.global_opcode()])?;

        Ok(())
    }
}

impl<SC: StarkProtocolConfig> VmCircuitExtension<SC> for Sha2 {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let bitwise_lu = {
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

        // this bus will be used for communication between the block hasher chip and the main chip
        let sha2_bus_index = inventory.new_bus_idx();
        // the sha2 subair needs its own bus for self-interactions
        let subair_bus_index = inventory.new_bus_idx();

        // SHA-256
        let sha256_block_hasher_air =
            Sha2BlockHasherVmAir::<Sha256Config>::new(bitwise_lu, subair_bus_index, sha2_bus_index);
        inventory.add_air(sha256_block_hasher_air);

        let sha256_main_air = Sha2MainAir::<Sha256Config>::new(
            inventory.system().port(),
            bitwise_lu,
            inventory.pointer_max_bits(),
            sha2_bus_index,
            Sha2Opcode::CLASS_OFFSET,
        );
        inventory.add_air(sha256_main_air);

        // SHA-512
        let sha512_block_hasher_air =
            Sha2BlockHasherVmAir::<Sha512Config>::new(bitwise_lu, subair_bus_index, sha2_bus_index);
        inventory.add_air(sha512_block_hasher_air);

        let sha512_main_air = Sha2MainAir::<Sha512Config>::new(
            inventory.system().port(),
            bitwise_lu,
            inventory.pointer_max_bits(),
            sha2_bus_index,
            Sha2Opcode::CLASS_OFFSET,
        );
        inventory.add_air(sha512_main_air);

        Ok(())
    }
}

pub struct Sha2CpuProverExt;
// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, Sha2> for Sha2CpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>> + Send + Sync + 'static,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        _: &Sha2,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let pointer_max_bits = inventory.airs().pointer_max_bits();

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

        // We must add each block hasher chip before the main chip to ensure that main chip does its
        // tracegen first, because the main chip will pass the records to the block hasher chip
        // after its tracegen is done.

        // SHA-256
        inventory.next_air::<Sha2BlockHasherVmAir<Sha256Config>>()?;
        // shared records between the main chip and the block hasher chip
        let records = Arc::new(Mutex::new(None));
        let sha256_block_hasher_chip = Sha2BlockHasherChip::<Val<SC>, Sha256Config>::new(
            bitwise_lu.clone(),
            pointer_max_bits,
            mem_helper.clone(),
            records.clone(),
        );
        inventory.add_periphery_chip(sha256_block_hasher_chip);

        inventory.next_air::<Sha2MainAir<Sha256Config>>()?;
        let sha256_main_chip = Sha2MainChip::<Val<SC>, Sha256Config>::new(
            records,
            bitwise_lu.clone(),
            pointer_max_bits,
            mem_helper.clone(),
        );
        inventory.add_executor_chip(sha256_main_chip);

        // SHA-512
        inventory.next_air::<Sha2BlockHasherVmAir<Sha512Config>>()?;
        // shared records between the main chip and the block hasher chip
        let records = Arc::new(Mutex::new(None));
        let sha512_block_hasher_chip = Sha2BlockHasherChip::<Val<SC>, Sha512Config>::new(
            bitwise_lu.clone(),
            pointer_max_bits,
            mem_helper.clone(),
            records.clone(),
        );
        inventory.add_periphery_chip(sha512_block_hasher_chip);

        inventory.next_air::<Sha2MainAir<Sha512Config>>()?;
        let sha512_main_chip = Sha2MainChip::<Val<SC>, Sha512Config>::new(
            records,
            bitwise_lu.clone(),
            pointer_max_bits,
            mem_helper.clone(),
        );
        inventory.add_executor_chip(sha512_main_chip);

        Ok(())
    }
}
