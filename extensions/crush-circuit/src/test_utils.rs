use openvm_circuit::arch::testing::{TestBuilder, memory::gen_pointer};
use openvm_instructions::{VmOpcode, instruction::Instruction};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rand::{Rng, rngs::StdRng};

use super::adapters::{RV_IS_TYPE_IMM_BITS, RV32_REGISTER_NUM_LIMBS};

// Returns (instruction, rd)
#[allow(dead_code)]
pub fn rv32_rand_write_register_or_imm<const NUM_LIMBS: usize>(
    tester: &mut impl TestBuilder<BabyBear>,
    rs1_writes: [u8; NUM_LIMBS],
    rs2_writes: [u8; NUM_LIMBS],
    imm: Option<usize>,
    opcode_with_offset: usize,
    rng: &mut StdRng,
) -> (Instruction<BabyBear>, usize) {
    let rs2_is_imm = imm.is_some();

    let rs1 = gen_pointer(rng, NUM_LIMBS);
    let rs2 = imm.unwrap_or_else(|| gen_pointer(rng, NUM_LIMBS));
    let rd = gen_pointer(rng, NUM_LIMBS);

    tester.write::<NUM_LIMBS>(1, rs1, rs1_writes.map(BabyBear::from_u8));
    if !rs2_is_imm {
        tester.write::<NUM_LIMBS>(1, rs2, rs2_writes.map(BabyBear::from_u8));
    }

    (
        Instruction::from_usize(
            VmOpcode::from_usize(opcode_with_offset),
            [rd, rs1, rs2, 1, if rs2_is_imm { 0 } else { 1 }],
        ),
        rd,
    )
}

#[allow(dead_code)]
pub fn generate_rv32_is_type_immediate(rng: &mut StdRng) -> (usize, [u8; RV32_REGISTER_NUM_LIMBS]) {
    let mut imm: u32 = rng.random_range(0..(1 << RV_IS_TYPE_IMM_BITS));
    if (imm & 0x800) != 0 {
        imm |= !0xFFF
    }
    (
        (imm & 0xFFFFFF) as usize,
        [
            imm as u8,
            (imm >> 8) as u8,
            (imm >> 16) as u8,
            (imm >> 16) as u8,
        ],
    )
}
