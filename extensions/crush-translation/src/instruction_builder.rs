use openvm_crush_transpiler::{
    BaseAlu64Opcode, BaseAluOpcode, CallOpcode, ConstOpcodes, Eq64Opcode, EqOpcode,
    HintStoreOpcode, JumpOpcode, Keccak256Opcode, LessThan64Opcode, LessThanOpcode, MulOpcode,
    Phantom, Shift64Opcode, ShiftOpcode,
};
use openvm_instructions::{LocalOpcode, SystemOpcode, VmOpcode, instruction::Instruction, riscv};
use openvm_stark_backend::p3_field::PrimeField32;

use openvm_rv32im_transpiler::Rv32LoadStoreOpcode as LoadStoreOpcode;

use crate::{ERROR_ABORT_CODE, ERROR_CODE_OFFSET};

/// Immediate in the format expected by the ALU adapter.
#[derive(Debug, Clone, Copy)]
pub struct AluImm(u32);

impl From<i16> for AluImm {
    fn from(value: i16) -> Self {
        // ALU adapter expects an i16 sign-extended to 24 bits (top byte zeroed).
        let value = value as i32 as u32 & 0xff_ff_ff;
        AluImm(value)
    }
}

impl TryFrom<u32> for AluImm {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        i16::try_from(value as i32)
            .map(AluImm::from)
            .map_err(|_| ())
    }
}

/// Build an R-type instruction (register-register-register).
/// Used by ALU, shift, comparison, and equality instructions.
/// All register indices are scaled by RV32_REGISTER_NUM_LIMBS (4 bytes per register).
/// d=1 and e=1 indicate both operands are read from register address space.
pub fn instr_r<F: PrimeField32>(
    opcode: usize,
    rd: usize,
    rs1: usize,
    rs2: usize,
) -> Instruction<F> {
    Instruction::new(
        VmOpcode::from_usize(opcode),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // a: destination register
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: source register 1
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs2), // c: source register 2
        F::ONE,                                                       // d: write AS (register)
        F::ONE,                                                       // e: read AS (register)
        F::ZERO,                                                      // f: (not used)
        F::ZERO,                                                      // g: (not used)
    )
}

/// Build an I-type instruction (register-register-immediate).
/// Used by ALU, shift, comparison, and equality instructions with an immediate operand.
/// d=1 selects register AS for the destination, e=0 signals immediate mode for the operand.
pub fn instr_i<F: PrimeField32>(
    opcode: usize,
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    let imm: AluImm = imm.into();
    Instruction::new(
        VmOpcode::from_usize(opcode),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // a: destination register
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: source register 1
        F::from_canonical_u32(imm.0),                                 // c: AluImm-encoded imm
        F::ONE,                                                       // d: write AS (register)
        F::ZERO,                                                      // e: read AS (0=immediate)
        F::ZERO,                                                      // f: (not used)
        F::ZERO,                                                      // g: (not used)
    )
}

// ---- 32-bit Arithmetic (BaseAluOpcode) ----

/// rd = rs1 + rs2 (wrapping, 32-bit)
pub fn add<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::ADD.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = rs1 + imm (wrapping, 32-bit)
#[allow(dead_code)]
pub fn add_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(BaseAluOpcode::ADD.global_opcode().as_usize(), rd, rs1, imm)
}

/// rd = rs1 - rs2 (wrapping, 32-bit)
pub fn sub<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::SUB.global_opcode().as_usize(), rd, rs1, rs2)
}

// ---- Multiplication ----

/// rd = (rs1 * rs2)[31:0] (low 32 bits)
pub fn mul<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(MulOpcode::MUL.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = (rs1 * imm)[31:0] (low 32 bits)
pub fn mul_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(MulOpcode::MUL.global_opcode().as_usize(), rd, rs1, imm)
}

/// rd = (rs1 * rs2)[63:0] (low 64 bits)
pub fn mul_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_crush_transpiler::Mul64Opcode;
    instr_r(Mul64Opcode::MUL.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = (rs1 * imm)[63:0] (low 64 bits)
pub fn mul_imm_64<F: PrimeField32>(rd: usize, rs1: usize, imm: AluImm) -> Instruction<F> {
    use openvm_crush_transpiler::Mul64Opcode;
    instr_i(Mul64Opcode::MUL.global_opcode().as_usize(), rd, rs1, imm)
}

// ---- Division / Remainder (32-bit) ----
// The circuit constraints follow the RISC-V spec for division by zero, which
// differs from WebAssembly semantics. See https://github.com/powdr-labs/powdr-wasm/issues/24

/// rd = rs1 /s rs2 (signed division, 32-bit)
pub fn div<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_crush_transpiler::DivRemOpcode;
    instr_r(DivRemOpcode::DIV.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = rs1 /u rs2 (unsigned division, 32-bit)
pub fn divu<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_crush_transpiler::DivRemOpcode;
    instr_r(DivRemOpcode::DIVU.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = rs1 %s rs2 (signed remainder, 32-bit)
pub fn rems<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_crush_transpiler::DivRemOpcode;
    instr_r(DivRemOpcode::REM.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = rs1 %u rs2 (unsigned remainder, 32-bit)
pub fn remu<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_crush_transpiler::DivRemOpcode;
    instr_r(DivRemOpcode::REMU.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = rs1 /s imm (signed division by immediate, 32-bit)
pub fn div_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    use openvm_crush_transpiler::DivRemOpcode;
    instr_i(DivRemOpcode::DIV.global_opcode().as_usize(), rd, rs1, imm)
}

/// rd = rs1 /u imm (unsigned division by immediate, 32-bit)
pub fn divu_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    use openvm_crush_transpiler::DivRemOpcode;
    instr_i(DivRemOpcode::DIVU.global_opcode().as_usize(), rd, rs1, imm)
}

/// rd = rs1 %s imm (signed remainder by immediate, 32-bit)
pub fn rems_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    use openvm_crush_transpiler::DivRemOpcode;
    instr_i(DivRemOpcode::REM.global_opcode().as_usize(), rd, rs1, imm)
}

/// rd = rs1 %u imm (unsigned remainder by immediate, 32-bit)
pub fn remu_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    use openvm_crush_transpiler::DivRemOpcode;
    instr_i(DivRemOpcode::REMU.global_opcode().as_usize(), rd, rs1, imm)
}

// ---- Division / Remainder (64-bit) ----

/// rd = rs1 /s rs2 (signed division, 64-bit)
pub fn div_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_crush_transpiler::DivRem64Opcode;
    instr_r(DivRem64Opcode::DIV.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = rs1 /u rs2 (unsigned division, 64-bit)
pub fn divu_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_crush_transpiler::DivRem64Opcode;
    instr_r(
        DivRem64Opcode::DIVU.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

/// rd = rs1 %s rs2 (signed remainder, 64-bit)
pub fn rems_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_crush_transpiler::DivRem64Opcode;
    instr_r(DivRem64Opcode::REM.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = rs1 %u rs2 (unsigned remainder, 64-bit)
pub fn remu_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    use openvm_crush_transpiler::DivRem64Opcode;
    instr_r(
        DivRem64Opcode::REMU.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

/// rd = rs1 /s imm (signed division by immediate, 64-bit)
pub fn div_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    use openvm_crush_transpiler::DivRem64Opcode;
    instr_i(DivRem64Opcode::DIV.global_opcode().as_usize(), rd, rs1, imm)
}

/// rd = rs1 /u imm (unsigned division by immediate, 64-bit)
pub fn divu_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    use openvm_crush_transpiler::DivRem64Opcode;
    instr_i(
        DivRem64Opcode::DIVU.global_opcode().as_usize(),
        rd,
        rs1,
        imm,
    )
}

/// rd = rs1 %s imm (signed remainder by immediate, 64-bit)
pub fn rems_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    use openvm_crush_transpiler::DivRem64Opcode;
    instr_i(DivRem64Opcode::REM.global_opcode().as_usize(), rd, rs1, imm)
}

/// rd = rs1 %u imm (unsigned remainder by immediate, 64-bit)
pub fn remu_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    use openvm_crush_transpiler::DivRem64Opcode;
    instr_i(
        DivRem64Opcode::REMU.global_opcode().as_usize(),
        rd,
        rs1,
        imm,
    )
}

/// rd = rs1 ^ rs2 (32-bit)
pub fn xor<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::XOR.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = rs1 | rs2 (32-bit)
pub fn or<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::OR.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = rs1 & rs2 (32-bit)
pub fn and<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAluOpcode::AND.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = rs1 & imm (32-bit)
pub fn and_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(BaseAluOpcode::AND.global_opcode().as_usize(), rd, rs1, imm)
}

// ---- Shifts (32-bit) ----

/// rd = rs1 << rs2 (32-bit)
pub fn shl<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(ShiftOpcode::SLL.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = rs1 << imm (32-bit)
pub fn shl_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(ShiftOpcode::SLL.global_opcode().as_usize(), rd, rs1, imm)
}

/// rd = rs1 >>u rs2 (logical shift right, 32-bit)
pub fn shr_u<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(ShiftOpcode::SRL.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = rs1 >>u imm (logical shift right, 32-bit)
pub fn shr_u_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(ShiftOpcode::SRL.global_opcode().as_usize(), rd, rs1, imm)
}

/// rd = rs1 >>s imm (arithmetic shift right, 32-bit)
pub fn shr_s_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(ShiftOpcode::SRA.global_opcode().as_usize(), rd, rs1, imm)
}

// ---- Shifts (64-bit) ----

/// rd = rs1 << rs2 (64-bit)
pub fn shl_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(Shift64Opcode::SLL.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = rs1 << imm (64-bit)
pub fn shl_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    instr_i(Shift64Opcode::SLL.global_opcode().as_usize(), rd, rs1, imm)
}

/// rd = rs1 >>u rs2 (logical shift right, 64-bit)
pub fn shr_u_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(Shift64Opcode::SRL.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = rs1 >>s imm (arithmetic shift right, 64-bit)
pub fn shr_s_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    instr_i(Shift64Opcode::SRA.global_opcode().as_usize(), rd, rs1, imm)
}

/// rd = rs1 >>u imm (logical shift right, 64-bit)
pub fn shr_u_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    instr_i(Shift64Opcode::SRL.global_opcode().as_usize(), rd, rs1, imm)
}

// ---- Comparisons (32-bit) ----

/// rd = (rs1 <u rs2) ? 1 : 0 (unsigned, 32-bit)
pub fn lt_u<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        LessThanOpcode::SLTU.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

/// rd = (rs1 <u imm) ? 1 : 0 (unsigned, 32-bit)
pub fn lt_u_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(
        LessThanOpcode::SLTU.global_opcode().as_usize(),
        rd,
        rs1,
        imm,
    )
}

/// rd = (rs1 <s rs2) ? 1 : 0 (signed, 32-bit)
pub fn lt_s<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(LessThanOpcode::SLT.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = (rs1 >u rs2) ? 1 : 0 — emitted as lt_u with swapped operands
pub fn gt_u<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    lt_u(rd, rs2, rs1)
}

/// rd = (rs1 >s rs2) ? 1 : 0 — emitted as lt_s with swapped operands
pub fn gt_s<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    lt_s(rd, rs2, rs1)
}

// ---- Comparisons (64-bit) ----

/// rd = (rs1 <u rs2) ? 1 : 0 (unsigned, 64-bit)
pub fn lt_u_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        LessThan64Opcode::SLTU.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

/// rd = (rs1 <s rs2) ? 1 : 0 (signed, 64-bit)
pub fn lt_s_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        LessThan64Opcode::SLT.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

/// rd = (rs1 >u rs2) ? 1 : 0 — emitted as lt_u_64 with swapped operands
pub fn gt_u_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    lt_u_64(rd, rs2, rs1)
}

/// rd = (rs1 >s rs2) ? 1 : 0 — emitted as lt_s_64 with swapped operands
pub fn gt_s_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    lt_s_64(rd, rs2, rs1)
}

// ---- Equality (32-bit) ----

/// rd = (rs1 == rs2) ? 1 : 0 (32-bit)
pub fn eq<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(EqOpcode::EQ.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = (rs1 == imm) ? 1 : 0 (32-bit)
pub fn eq_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(EqOpcode::EQ.global_opcode().as_usize(), rd, rs1, imm)
}

/// rd = (rs1 != rs2) ? 1 : 0 (32-bit)
pub fn neq<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(EqOpcode::NEQ.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = (rs1 != imm) ? 1 : 0 (32-bit)
pub fn neq_imm<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(EqOpcode::NEQ.global_opcode().as_usize(), rd, rs1, imm)
}

// ---- Equality (64-bit) ----

/// rd = (rs1 == rs2) ? 1 : 0 (64-bit)
pub fn eq_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(Eq64Opcode::EQ.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = (rs1 == imm) ? 1 : 0 (64-bit)
pub fn eq_imm_64<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(Eq64Opcode::EQ.global_opcode().as_usize(), rd, rs1, imm)
}

/// rd = (rs1 != rs2) ? 1 : 0 (64-bit)
pub fn neq_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(Eq64Opcode::NEQ.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = (rs1 != imm) ? 1 : 0 (64-bit)
pub fn neq_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    instr_i(Eq64Opcode::NEQ.global_opcode().as_usize(), rd, rs1, imm)
}

// ---- Constant Loading ----

/// CONST32: Load a 32-bit immediate into a register.
/// The immediate is split into two 16-bit halves: target_reg = (imm_hi << 16) | imm_lo.
pub fn const_32_imm<F: PrimeField32>(
    target_reg: usize,
    imm_lo: u16,
    imm_hi: u16,
) -> Instruction<F> {
    Instruction::new(
        ConstOpcodes::CONST32.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * target_reg), // a: target_reg
        F::from_canonical_usize(imm_lo as usize),                             // b: imm low 16b
        F::from_canonical_usize(imm_hi as usize),                             // c: imm high 16b
        F::ZERO,                                                              // d: (not used)
        F::ZERO,                                                              // e: (not used)
        F::ONE,                                                               // f: (not used)
        F::ZERO,                                                              // g: (not used)
    )
}

// ---- 64-bit Arithmetic (BaseAlu64Opcode) ----

/// rd = rs1 + rs2 (wrapping, 64-bit)
pub fn add_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        BaseAlu64Opcode::ADD.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

/// rd = rs1 + imm (wrapping, 64-bit)
pub fn add_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    instr_i(
        BaseAlu64Opcode::ADD.global_opcode().as_usize(),
        rd,
        rs1,
        imm,
    )
}

/// rd = rs1 - rs2 (wrapping, 64-bit)
pub fn sub_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        BaseAlu64Opcode::SUB.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

/// rd = rs1 - imm (wrapping, 64-bit)
pub fn sub_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    instr_i(
        BaseAlu64Opcode::SUB.global_opcode().as_usize(),
        rd,
        rs1,
        imm,
    )
}

/// rd = rs1 ^ rs2 (64-bit)
pub fn xor_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        BaseAlu64Opcode::XOR.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

/// rd = rs1 ^ imm (64-bit)
pub fn xor_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    instr_i(
        BaseAlu64Opcode::XOR.global_opcode().as_usize(),
        rd,
        rs1,
        imm,
    )
}

/// rd = rs1 | rs2 (64-bit)
pub fn or_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(BaseAlu64Opcode::OR.global_opcode().as_usize(), rd, rs1, rs2)
}

/// rd = rs1 | imm (64-bit)
pub fn or_imm_64<F: PrimeField32>(rd: usize, rs1: usize, imm: impl Into<AluImm>) -> Instruction<F> {
    instr_i(BaseAlu64Opcode::OR.global_opcode().as_usize(), rd, rs1, imm)
}

/// rd = rs1 & rs2 (64-bit)
pub fn and_64<F: PrimeField32>(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    instr_r(
        BaseAlu64Opcode::AND.global_opcode().as_usize(),
        rd,
        rs1,
        rs2,
    )
}

/// rd = rs1 & imm (64-bit)
pub fn and_imm_64<F: PrimeField32>(
    rd: usize,
    rs1: usize,
    imm: impl Into<AluImm>,
) -> Instruction<F> {
    instr_i(
        BaseAlu64Opcode::AND.global_opcode().as_usize(),
        rd,
        rs1,
        imm,
    )
}

// ---- Call / Return ----

/// RET: Restore PC and FP from registers previously saved by CALL/CALL_INDIRECT.
/// Reads the saved PC and absolute FP from registers relative to the current FP.
pub fn ret<F: PrimeField32>(to_pc_reg: usize, to_fp_reg: usize) -> Instruction<F> {
    Instruction::new(
        CallOpcode::RET.global_opcode(),
        F::ZERO,                                                             // a: (not used)
        F::ZERO,                                                             // b: (not used)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * to_pc_reg), // c: to_pc_operand
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * to_fp_reg), // d: to_fp_operand
        F::ONE,                                                              // e: PC read AS
        F::ONE,                                                              // f: FP read AS
        F::ZERO,                                                             // g: (unused)
    )
}

/// CALL instruction: Call function (save PC and FP, jump to label)
/// Saves current PC and FP, then sets PC from immediate and FP = current_FP + fp_offset
pub fn call<F: PrimeField32>(
    save_pc: usize,
    save_fp: usize,
    to_pc_imm: usize,
    fp_offset: usize,
) -> Instruction<F> {
    Instruction::new(
        CallOpcode::CALL.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * save_pc), // a: rd1 (save PC here)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * save_fp), // b: rd2 (save FP here)
        F::from_canonical_usize(to_pc_imm), // c: to_pc_operand (immediate PC target)
        F::from_canonical_usize(fp_offset), // d: to_fp_operand (FP offset)
        F::ZERO,                            // e: PC read AS (0 = no register read)
        F::ZERO,                            // f: FP read AS (0 = no register read)
        F::ZERO,                            // g: (unused)
    )
}

/// CALL_INDIRECT instruction: Call function indirect (save PC and FP, jump to register)
/// Saves current PC and FP, then sets PC from register and FP = current_FP + fp_offset
#[allow(dead_code)]
pub fn call_indirect<F: PrimeField32>(
    save_pc: usize,
    save_fp: usize,
    to_pc_reg: usize,
    fp_offset: usize,
) -> Instruction<F> {
    Instruction::new(
        CallOpcode::CALL_INDIRECT.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * save_pc),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * save_fp),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * to_pc_reg), // c: to_pc_operand
        F::from_canonical_usize(fp_offset), // d: to_fp_operand (FP offset)
        F::ONE,                             // e: PC read AS (1 = register read)
        F::ZERO,                            // f: FP read AS (0 = no register read)
        F::ZERO,                            // g: (unused)
    )
}

// ---- Jumps ----

/// JUMP: Unconditional jump to immediate PC.
pub fn jump<F: PrimeField32>(to_pc_imm: usize) -> Instruction<F> {
    Instruction::new(
        JumpOpcode::JUMP.global_opcode(),
        F::from_canonical_usize(to_pc_imm), // a: to_pc_imm
        F::ZERO,                            // b: (not used)
        F::ZERO,                            // c: (not used)
        F::ZERO,                            // d: (not used)
        F::ZERO,                            // e: (not used)
        F::ONE,                             // f
        F::ZERO,                            // g: imm sign
    )
}

/// SKIP: Relative jump by register offset. PC += (offset + 1) * DEFAULT_PC_STEP.
/// The +1 accounts for crush's natural PC increment (offset=0 would otherwise loop forever).
pub fn skip<F: PrimeField32>(offset_reg: usize) -> Instruction<F> {
    Instruction::new(
        JumpOpcode::SKIP.global_opcode(),
        F::ZERO,                                                              // a: (not used)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * offset_reg), // b: register with the offset
        F::ZERO,                                                              // c: (not used)
        F::ZERO,                                                              // d: (not used)
        F::ZERO,                                                              // e: (not used)
        F::ONE,                                                               // f
        F::ZERO,                                                              // g: imm sign
    )
}

/// JUMP_IF: Jump to immediate PC if condition register is non-zero.
pub fn jump_if<F: PrimeField32>(condition_reg: usize, to_pc_imm: usize) -> Instruction<F> {
    Instruction::new(
        JumpOpcode::JUMP_IF.global_opcode(),
        F::from_canonical_usize(to_pc_imm), // a: to_pc_imm
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * condition_reg), // b: condition_reg
        F::ZERO,                            // c: (not used)
        F::ZERO,                            // d: (not used)
        F::ZERO,                            // e: (not used)
        F::ONE,                             // f
        F::ZERO,                            // g: imm sign
    )
}

/// JUMP_IF_ZERO: Jump to immediate PC if condition register is zero.
pub fn jump_if_zero<F: PrimeField32>(condition_reg: usize, to_pc_imm: usize) -> Instruction<F> {
    Instruction::new(
        JumpOpcode::JUMP_IF_ZERO.global_opcode(),
        F::from_canonical_usize(to_pc_imm), // a: to_pc_imm
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * condition_reg), // b: condition_reg
        F::ZERO,                            // c: (not used)
        F::ZERO,                            // d: (not used)
        F::ZERO,                            // e: (not used)
        F::ONE,                             // f
        F::ZERO,                            // g: imm sign
    )
}

// ---- Memory Load/Store ----
// All load/store instructions use base register + immediate offset addressing.
// The immediate is split: lower 16 bits in field c, upper 16 bits in field g.

/// LOADW: rd = MEM32[rs1 + imm]
pub fn loadw<F: PrimeField32>(rd: usize, rs1: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::LOADW.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // a: rd
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1
        F::from_canonical_u32(imm & 0xFFFF),                          // c: imm (lower 16 bits)
        F::ONE,                                                       // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for word)
        F::ONE,                     // f
        F::from_canonical_u32(imm >> 16), // g: imm (higher 16 bits)
    )
}

/// STOREW: MEM32[rs1 + imm] = rs2
pub fn storew<F: PrimeField32>(value: usize, base_address: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::STOREW.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * value), // a: rs2 (data to store)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * base_address), // b: rs1 (base address)
        F::from_canonical_u32(imm & 0xFFFF), // c: imm (lower 16 bits)
        F::ONE,                              // d: register address space
        F::from_canonical_usize(2),          // e: memory address space (2 for word)
        F::ONE,                              // f
        F::from_canonical_u32(imm >> 16),    // g: imm (higher 16 bits)
    )
}

/// LOADB: rd = sign_extend(MEM8[rs1 + imm])
pub fn loadb<F: PrimeField32>(rd: usize, rs1: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::LOADB.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // a: rd
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1
        F::from_canonical_u32(imm & 0xFFFF),                          // c: imm (lower 16 bits)
        F::ONE,                                                       // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for byte)
        F::ONE,                     // f
        F::from_canonical_u32(imm >> 16), // g: imm (higher 16 bits)
    )
}

/// LOADBU: rd = zero_extend(MEM8[rs1 + imm])
pub fn loadbu<F: PrimeField32>(rd: usize, rs1: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::LOADBU.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // a: rd
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1
        F::from_canonical_u32(imm & 0xFFFF),                          // c: imm (lower 16 bits)
        F::ONE,                                                       // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for byte)
        F::ONE,                     // f
        F::from_canonical_u32(imm >> 16), // g: imm (higher 16 bits)
    )
}

/// LOADH: rd = sign_extend(MEM16[rs1 + imm])
#[allow(unused)]
pub fn loadh<F: PrimeField32>(rd: usize, rs1: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::LOADH.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // a: rd
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1
        F::from_canonical_u32(imm & 0xFFFF),                          // c: imm (lower 16 bits)
        F::ONE,                                                       // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for halfword)
        F::ONE,                     // f
        F::from_canonical_u32(imm >> 16), // g: imm (higher 16 bits)
    )
}

/// LOADHU: rd = zero_extend(MEM16[rs1 + imm])
pub fn loadhu<F: PrimeField32>(rd: usize, rs1: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::LOADHU.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd), // a: rd
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1
        F::from_canonical_u32(imm & 0xFFFF),                          // c: imm (lower 16 bits)
        F::ONE,                                                       // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for halfword)
        F::ONE,                     // f
        F::from_canonical_u32(imm >> 16), // g: imm (higher 16 bits)
    )
}

/// STOREB: MEM8[rs1 + imm] = rs2[7:0]
pub fn storeb<F: PrimeField32>(rs2: usize, rs1: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::STOREB.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs2), // a: rs2 (data to store)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1 (base address)
        F::from_canonical_u32(imm & 0xFFFF),                           // c: imm (lower 16 bits)
        F::ONE,                                                        // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for byte)
        F::ONE,                     // f
        F::from_canonical_u32(imm >> 16), // g: imm (higher 16 bits)
    )
}

/// STOREH: MEM16[rs1 + imm] = rs2[15:0]
pub fn storeh<F: PrimeField32>(rs2: usize, rs1: usize, imm: u32) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::STOREH.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs2), // a: rs2 (data to store)
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1), // b: rs1 (base address)
        F::from_canonical_u32(imm & 0xFFFF),                           // c: imm (lower 16 bits)
        F::ONE,                                                        // d: register address space
        F::from_canonical_usize(2), // e: memory address space (2 for halfword)
        F::ONE,                     // f
        F::from_canonical_u32(imm >> 16), // g: imm (higher 16 bits)
    )
}

// ---- Public Output ----

/// Reveal: Write register value to public output area (PUBLIC_VALUES_AS = 3).
/// Uses STOREW with memory AS overridden to the public values address space.
#[allow(dead_code)]
pub fn reveal<F: PrimeField32>(rs1_data: usize, rd_index: usize) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::STOREW.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rs1_data),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * rd_index),
        F::ZERO,
        F::ONE,
        F::from_canonical_usize(3),
        F::ONE,
        F::ZERO,
    )
}

/// Reveal with immediate output index offset.
#[allow(dead_code)]
pub fn reveal_imm<F: PrimeField32>(
    data_reg: usize,
    output_index_reg: usize,
    output_index_imm: usize,
) -> Instruction<F> {
    Instruction::new(
        LoadStoreOpcode::STOREW.global_opcode(),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * data_reg),
        F::from_canonical_usize(riscv::RV32_REGISTER_NUM_LIMBS * output_index_reg),
        F::from_canonical_usize(output_index_imm),
        F::ONE,
        F::from_canonical_usize(3),
        F::ONE,
        F::ZERO,
    )
}

// ---- System Instructions ----

/// TRAP: Terminate with exit code = ERROR_CODE_OFFSET (100) + error_code.
/// Used for WebAssembly traps (unreachable, out-of-bounds, etc.).
pub fn trap<F: PrimeField32>(error_code: usize) -> Instruction<F> {
    Instruction::new(
        SystemOpcode::TERMINATE.global_opcode(),
        F::ZERO,
        F::ZERO,
        F::from_canonical_usize(ERROR_CODE_OFFSET as usize + error_code),
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    )
}

/// ABORT: Terminate with exit code = ERROR_ABORT_CODE (200).
pub fn abort<F: PrimeField32>() -> Instruction<F> {
    Instruction::new(
        SystemOpcode::TERMINATE.global_opcode(),
        F::ZERO,
        F::ZERO,
        F::from_canonical_usize(ERROR_ABORT_CODE as usize),
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    )
}

/// HALT: Normal program termination (exit code = 0).
#[allow(dead_code)]
pub fn halt<F: PrimeField32>() -> Instruction<F> {
    Instruction::new(
        SystemOpcode::TERMINATE.global_opcode(),
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    )
}

// ---- Phantom / Hint Instructions ----

/// HintInput phantom: Pop next input vector, prepend its 4-byte LE length, push onto hint stream.
pub fn prepare_read<F: PrimeField32>() -> Instruction<F> {
    Instruction::new(
        SystemOpcode::PHANTOM.global_opcode(),
        F::ZERO,
        F::ZERO,
        F::from_canonical_u32(Phantom::HintInput as u32),
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    )
}

/// PrintStr phantom: Read string from memory at [mem_ptr_reg + mem_imm] and print to stdout.
pub fn debug_print<F: PrimeField32>(
    mem_ptr_reg: usize,
    amount_reg: usize,
    mem_imm: u16,
) -> Instruction<F> {
    // OpenVM execution splits c into c_hi and c_lo which are passed separately to the trait impl.
    let c = ((mem_imm as usize) << 16) | (Phantom::PrintStr as usize);
    Instruction::from_isize(
        SystemOpcode::PHANTOM.global_opcode(),
        (riscv::RV32_REGISTER_NUM_LIMBS * mem_ptr_reg) as isize,
        (riscv::RV32_REGISTER_NUM_LIMBS * amount_reg) as isize,
        c as isize,
        0,
        0,
    )
}

/// HINT_STOREW: Read one word from hint stream and write to MEM[reg[mem_ptr_reg]]
pub fn hint_storew<F: PrimeField32>(mem_ptr_reg: usize) -> Instruction<F> {
    Instruction::from_isize(
        HintStoreOpcode::HINT_STOREW.global_opcode(),
        0,
        (riscv::RV32_REGISTER_NUM_LIMBS * mem_ptr_reg) as isize,
        0,
        riscv::RV32_REGISTER_AS as isize,
        riscv::RV32_MEMORY_AS as isize,
    )
}

/// HINT_BUFFER: Read num_words words from hint stream and write to MEM[reg[mem_ptr_reg]..]
pub fn hint_buffer<F: PrimeField32>(num_words_reg: usize, mem_ptr_reg: usize) -> Instruction<F> {
    Instruction::from_isize(
        HintStoreOpcode::HINT_BUFFER.global_opcode(),
        (riscv::RV32_REGISTER_NUM_LIMBS * num_words_reg) as isize,
        (riscv::RV32_REGISTER_NUM_LIMBS * mem_ptr_reg) as isize,
        0,
        riscv::RV32_REGISTER_AS as isize,
        riscv::RV32_MEMORY_AS as isize,
    )
}

/// KECCAK256: Compute keccak256 hash of variable-length input.
/// dst_reg: register containing pointer to 32-byte output buffer
/// src_reg: register containing pointer to input data
/// len_reg: register containing input length in bytes
pub fn keccak256<F: PrimeField32>(
    dst_reg: usize,
    src_reg: usize,
    len_reg: usize,
) -> Instruction<F> {
    Instruction::from_isize(
        Keccak256Opcode::KECCAK256.global_opcode(),
        (riscv::RV32_REGISTER_NUM_LIMBS * dst_reg) as isize,
        (riscv::RV32_REGISTER_NUM_LIMBS * src_reg) as isize,
        (riscv::RV32_REGISTER_NUM_LIMBS * len_reg) as isize,
        riscv::RV32_REGISTER_AS as isize,
        riscv::RV32_MEMORY_AS as isize,
    )
}

/// TraceSyscall phantom: Prints "[wasi] #<seq> <name>" to stderr.
/// c_upper encodes a syscall ID that the phantom maps to a name string.
pub fn trace_syscall<F: PrimeField32>(syscall_id: u16) -> Instruction<F> {
    let c = ((syscall_id as usize) << 16) | (Phantom::TraceSyscall as usize);
    Instruction::from_isize(
        SystemOpcode::PHANTOM.global_opcode(),
        0,
        0,
        c as isize,
        0,
        0,
    )
}

/// ClockTimeGet phantom: Fills the hint stream with 8 bytes of incrementing timestamp.
pub fn clock_time_get<F: PrimeField32>() -> Instruction<F> {
    Instruction::from_isize(
        SystemOpcode::PHANTOM.global_opcode(),
        0,
        0,
        Phantom::ClockTimeGet as isize,
        0,
        0,
    )
}

/// HintRandom phantom: Fills the hint stream with `len` random words (4 bytes each).
/// `num_words_reg` is the register holding the number of words to generate.
pub fn prepare_random<F: PrimeField32>(num_words_reg: usize) -> Instruction<F> {
    Instruction::from_isize(
        SystemOpcode::PHANTOM.global_opcode(),
        (riscv::RV32_REGISTER_NUM_LIMBS * num_words_reg) as isize,
        0,
        Phantom::HintRandom as isize,
        0,
        0,
    )
}
