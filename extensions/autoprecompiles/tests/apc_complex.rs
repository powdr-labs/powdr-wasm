mod common;

use crush_translation::instruction_builder as ib;
use openvm_instructions::instruction::Instruction;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use powdr_autoprecompiles::blocks::BasicBlock;

fn assert_machine_output(program: Vec<Instruction<BabyBear>>, test_name: &str) {
    let bb = BasicBlock {
        start_pc: 0,
        instructions: program,
    };
    common::assert_machine_output(bb.into(), "complex", test_name);
}

// ==================== LoadStore roundtrips ====================

#[test]
fn storeb_roundtrip() {
    assert_machine_output(
        vec![
            ib::storeb(0, 1, 0),
            ib::storeb(0, 1, 1),
            ib::loadbu(2, 1, 0),
            ib::loadbu(3, 1, 1),
            ib::add(4, 2, 3),
        ],
        "storeb_roundtrip",
    );
}

#[test]
fn storeh_roundtrip() {
    assert_machine_output(
        vec![
            ib::storeh(0, 2, 0),
            ib::storeh(1, 2, 2),
            ib::loadhu(3, 2, 0),
            ib::loadhu(4, 2, 2),
            ib::add(5, 3, 4),
        ],
        "storeh_roundtrip",
    );
}

#[test]
fn storeb_loadb_roundtrip() {
    assert_machine_output(
        vec![ib::storeb(0, 1, 0x2_0004), ib::loadb(2, 1, 0x2_0004)],
        "storeb_loadb_roundtrip",
    );
}

// ==================== Comparison sequences ====================

#[test]
fn comparison_chain() {
    assert_machine_output(
        vec![ib::lt_u(3, 0, 1), ib::lt_u(4, 1, 2)],
        "comparison_chain",
    );
}

#[test]
fn mixed_comparisons() {
    assert_machine_output(
        vec![ib::gt_u(2, 0, 1), ib::gt_s(5, 3, 4)],
        "mixed_comparisons",
    );
}

#[test]
fn comparison_equivalence() {
    assert_machine_output(
        vec![ib::gt_u(2, 0, 1), ib::lt_u(3, 1, 0), ib::xor(4, 2, 3)],
        "comparison_equivalence",
    );
}

#[test]
fn mixed_signed_unsigned() {
    assert_machine_output(
        vec![ib::gt_u(2, 0, 1), ib::gt_s(3, 0, 1), ib::sub(4, 2, 3)],
        "mixed_signed_unsigned",
    );
}

// ==================== Mul sequences ====================

#[test]
fn mul_commutative() {
    assert_machine_output(
        vec![ib::mul(2, 0, 1), ib::mul(3, 1, 0), ib::sub(4, 2, 3)],
        "mul_commutative",
    );
}

#[test]
fn mul_chain() {
    assert_machine_output(vec![ib::mul(3, 0, 1), ib::mul(4, 3, 2)], "mul_chain");
}

// ==================== Div sequences ====================

#[test]
fn div_chain() {
    assert_machine_output(vec![ib::div(3, 0, 1), ib::div(4, 3, 2)], "div_chain");
}

#[test]
fn div_and_mul_inverse() {
    assert_machine_output(
        vec![ib::div(2, 0, 1), ib::mul(3, 2, 1)],
        "div_and_mul_inverse",
    );
}

// ==================== Cross-width ====================

#[test]
fn cross_width_32_to_64() {
    assert_machine_output(
        vec![ib::add_imm(0, 0, 0x42_i16), ib::add_imm_64(2, 0, 0_i16)],
        "cross_width_32_to_64",
    );
}

#[test]
fn cross_width_64_to_32() {
    assert_machine_output(
        vec![ib::add_imm_64(0, 0, 0x42_i16), ib::add_imm(2, 0, 0_i16)],
        "cross_width_64_to_32",
    );
}

// ==================== Memory copy / fill loops ====================
// Inner loops from musl memmove (memory.copy) and memset (memory.fill),
// compiled to WOMIR from the C builtins in integration/builtin_src/.

#[test]
fn copy_byte() {
    // Forward byte-by-byte copy loop body (musl memmove)
    // C: for (; n; n--) *d++ = *s++;
    assert_machine_output(
        vec![
            ib::loadbu(3, 0, 0),       // r3 = zero_extend(MEM8[r0])
            ib::storeb(3, 1, 0),       // MEM8[r1] = r3
            ib::add_imm(0, 0, 1_i16),  // r0++ (src advance)
            ib::add_imm(1, 1, 1_i16),  // r1++ (dst advance)
            ib::add_imm(2, 2, -1_i16), // r2-- (byte count)
        ],
        "copy_byte",
    );
}

#[test]
fn copy_word() {
    // Forward word-aligned copy loop body (musl memmove)
    // C: for (; n>=WS; n-=WS, d+=WS, s+=WS) *(WT *)d = *(WT *)s;
    assert_machine_output(
        vec![
            ib::loadw(3, 0, 0),        // r3 = MEM32[r0]
            ib::storew(3, 1, 0),       // MEM32[r1] = r3
            ib::add_imm(0, 0, 4_i16),  // r0 += 4 (src advance)
            ib::add_imm(1, 1, 4_i16),  // r1 += 4 (dst advance)
            ib::add_imm(2, 2, -4_i16), // r2 -= 4 (byte count)
        ],
        "copy_word",
    );
}

#[test]
fn fill_32bytes() {
    // 32-byte fill inner loop body (musl memset)
    // Fills 32 bytes per iteration using 8 word stores (4 × u64 = 8 × u32).
    // Uses intermediate ADD_IMMs to compute store addresses, matching the
    // actual WOMIR compilation of the memory_fill builtin.
    // C: for (; n >= 32; n-=32, s+=32) { *(u64*)(s+0) = c64; ... *(u64*)(s+24) = c64; }
    assert_machine_output(
        vec![
            ib::storew(5, 0, 0),        // MEM32[r0+0] = r5 (c32)
            ib::storew(6, 0, 4),        // MEM32[r0+4] = r6 (c32)
            ib::add_imm(2, 0, 24_i16),  // r2 = r0 + 24
            ib::storew(5, 2, 0),        // MEM32[r0+24] = r5
            ib::storew(6, 2, 4),        // MEM32[r0+28] = r6
            ib::add_imm(2, 0, 16_i16),  // r2 = r0 + 16
            ib::storew(5, 2, 0),        // MEM32[r0+16] = r5
            ib::storew(6, 2, 4),        // MEM32[r0+20] = r6
            ib::add_imm(2, 0, 8_i16),   // r2 = r0 + 8
            ib::storew(5, 2, 0),        // MEM32[r0+8] = r5
            ib::storew(6, 2, 4),        // MEM32[r0+12] = r6
            ib::add_imm(0, 0, 32_i16),  // r0 += 32 (advance pointer)
            ib::add_imm(1, 1, -32_i16), // r1 -= 32 (decrement remaining)
        ],
        "fill_32bytes",
    );
}


// ==================== I64Load patterns ====================

#[test]
fn i64load_output_plus_1_eq_base_addr() {
    assert_machine_output(
        vec![ib::loadw(0, 1, 0), ib::loadw(1, 1, 4)],
        "i64load_output_plus_1_eq_base_addr",
    );
}

#[test]
fn i64load_output_eq_base_addr() {
    assert_machine_output(
        vec![ib::loadw(1, 0, 4), ib::loadw(0, 0, 0)],
        "i64load_output_eq_base_addr",
    );
}

#[test]
fn i64load32u() {
    assert_machine_output(
        vec![ib::loadw(0, 1, 0), ib::const_32_imm(1, 0, 0)],
        "i64load32u",
    );
}

// ==================== Reproduce suboptimal APCs ====================

#[test]
fn same_register_not_optimized() {
    assert_machine_output(
        // Stores a constant in a register, then stores it into memory.
        //
        // This test showcases an inefficiency in the current combination of crush and the APC optimizer:
        // - In the load/store chip, the flags cannot be solved at compile time because they encode BOTH the
        //   opcode (known at compile time) and the alignment offset (only known at runtime).
        // - The flags change the address going to RAM, but not the address going to register memory.
        // - This is not removed though, the register address is:
        //   `from_state__fp_0 + 12 + <some expression depending on the flags>`
        // - In practice, the only valid assignment to the flags is such that the expression is 0, but the
        //   optimizer doesn't know that, so it leaves the flags as symbolic.
        // - When the second instruction accesses the same register as the first, the memory optimizer sees the symbolic
        //   flags and doesn't realize that the register address is actually the same as in the first instruction.
        vec![ib::const_32_imm(3, 123, 0), ib::storeb(3, 1, 0)],
        "same_register_not_optimized",
    );
}
