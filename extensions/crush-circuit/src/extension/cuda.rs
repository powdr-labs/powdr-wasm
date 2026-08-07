use std::sync::Arc;

use openvm_circuit::{
    arch::{ChipInventory, ChipInventoryError, DenseRecordArena, VmProverExtension},
    system::cuda::extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
};
use openvm_circuit_primitives::range_tuple::{RangeTupleCheckerAir, RangeTupleCheckerChipGPU};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine, GpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::{
    BaseAlu32Air, BaseAlu32ChipGpu, BaseAlu64Air, BaseAlu64ChipGpu, CallAir, CallChipGpu,
    Const32Air, Const32ChipGpu, DivRem32Air, DivRem32ChipGpu, DivRem64Air, DivRem64ChipGpu,
    Eq32Air, Eq32ChipGpu, Eq64Air, Eq64ChipGpu, HintStoreAir, HintStoreChipGpu, JumpAir,
    JumpChipGpu, LessThan32Air, LessThan32ChipGpu, LessThan64Air, LessThan64ChipGpu,
    LoadSignExtendAir, LoadSignExtendChipGpu, LoadStoreAir, LoadStoreChipGpu, Mul64Air,
    Mul64ChipGpu, Multiplication32Air, Multiplication32ChipGpu, Shift32Air, Shift32ChipGpu,
    Shift64Air, Shift64ChipGpu,
};

use super::Crush;

pub struct CrushGpuProverExt;

impl VmProverExtension<BabyBearPoseidon2GpuEngine, DenseRecordArena, Crush> for CrushGpuProverExt {
    fn extend_prover(
        &self,
        extension: &Crush,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let pointer_max_bits = inventory.airs().pointer_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        // NOTE: The order of next_air() calls must match the order of add_air()
        // calls in Crush::extend_circuit (extension/mod.rs).

        inventory.next_air::<BaseAlu32Air>()?;
        let base_alu = BaseAlu32ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(base_alu);

        inventory.next_air::<BaseAlu64Air>()?;
        let base_alu_64 = BaseAlu64ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(base_alu_64);

        let range_tuple_checker = {
            let existing_chip = inventory
                .find_chip::<Arc<RangeTupleCheckerChipGPU<2>>>()
                .find(|c| {
                    c.sizes[0] >= extension.range_tuple_checker_sizes[0]
                        && c.sizes[1] >= extension.range_tuple_checker_sizes[1]
                });
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                inventory.next_air::<RangeTupleCheckerAir<2>>()?;
                let chip = Arc::new(RangeTupleCheckerChipGPU::new(
                    extension.range_tuple_checker_sizes,
                ));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        inventory.next_air::<Multiplication32Air>()?;
        let mul = Multiplication32ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mul);

        inventory.next_air::<Mul64Air>()?;
        let mul_64 = Mul64ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mul_64);

        inventory.next_air::<LessThan32Air>()?;
        let less_than = LessThan32ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(less_than);

        inventory.next_air::<LessThan64Air>()?;
        let less_than_64 = LessThan64ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(less_than_64);

        inventory.next_air::<DivRem32Air>()?;
        let divrem = DivRem32ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(divrem);

        inventory.next_air::<DivRem64Air>()?;
        let divrem_64 = DivRem64ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(divrem_64);

        inventory.next_air::<Eq32Air>()?;
        let eq = Eq32ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(eq);

        inventory.next_air::<Eq64Air>()?;
        let eq_64 = Eq64ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(eq_64);

        inventory.next_air::<Shift32Air>()?;
        let shift = Shift32ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(shift);

        inventory.next_air::<Shift64Air>()?;
        let shift_64 = Shift64ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(shift_64);

        inventory.next_air::<LoadStoreAir>()?;
        let load_store =
            LoadStoreChipGpu::new(range_checker.clone(), pointer_max_bits, timestamp_max_bits);
        inventory.add_executor_chip(load_store);

        inventory.next_air::<LoadSignExtendAir>()?;
        let load_sign_extend =
            LoadSignExtendChipGpu::new(range_checker.clone(), pointer_max_bits, timestamp_max_bits);
        inventory.add_executor_chip(load_sign_extend);

        inventory.next_air::<JumpAir>()?;
        let jump = JumpChipGpu::new(range_checker.clone(), timestamp_max_bits);
        inventory.add_executor_chip(jump);

        inventory.next_air::<Const32Air>()?;
        let const32 = Const32ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            timestamp_max_bits,
        );
        inventory.add_executor_chip(const32);

        inventory.next_air::<CallAir>()?;
        let call = CallChipGpu::new(range_checker.clone(), pointer_max_bits, timestamp_max_bits);
        inventory.add_executor_chip(call);

        inventory.next_air::<HintStoreAir>()?;
        let hint_store = HintStoreChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(hint_store);

        Ok(())
    }
}
