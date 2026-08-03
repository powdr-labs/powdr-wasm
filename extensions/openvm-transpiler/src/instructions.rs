use openvm_instructions::LocalOpcode;
use openvm_instructions_derive::LocalOpcode;
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumIter, FromRepr, IntoEnumIterator};

pub use openvm_rv32im_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, DivRemOpcode, LessThanOpcode,
    MulOpcode, Rv32LoadStoreOpcode as LoadStoreOpcode, ShiftOpcode,
};

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x2200]
#[repr(usize)]
// Note: these need to be exactly the same and in the exact same order as OpenVM's BaseAluOpcode
// in order to be able to re-use the original Alu core chip.
// We do re-use the `BaseAluOpcode` type for 32-bit operations, but need a new opcode/enum for
// 64-bit ops.
pub enum BaseAlu64Opcode {
    ADD,
    SUB,
    XOR,
    OR,
    AND,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x2205]
#[repr(usize)]
// Note: these need to be exactly the same and in the exact same order as OpenVM's ShiftOpcode
// in order to be able to re-use the original Shift core chip.
// We do re-use the `BaseAluOpcode` type for 32-bit operations, but need a new opcode/enum for
// 64-bit ops.
pub enum Shift64Opcode {
    SLL,
    SRL,
    SRA,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x2208]
#[repr(usize)]
// Note: these need to be exactly the same and in the exact same order as OpenVM's LessThanOpcode
// in order to be able to re-use the original LessThan core chip.
// We do re-use the `BaseAluOpcode` type for 32-bit operations, but need a new opcode/enum for
// 64-bit ops.
pub enum LessThan64Opcode {
    SLT,
    SLTU,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x120c]
#[repr(usize)]
pub enum EqOpcode {
    EQ,
    NEQ,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x220c]
#[repr(usize)]
pub enum Eq64Opcode {
    EQ,
    NEQ,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x1236]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum CallOpcode {
    RET,           // return to pc and restore frame
    CALL,          // call function, save pc and fp
    CALL_INDIRECT, // call function indirect, save pc and fp
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x123B]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum JumpOpcode {
    JUMP,         // unconditional jump to immediate PC
    SKIP,         // unconditional jump to current PC + offset
    JUMP_IF,      // conditional jump to immediate PC if condition register != 0
    JUMP_IF_ZERO, // conditional jump to immediate PC if condition register == 0
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x2250]
#[repr(usize)]
#[allow(non_camel_case_types)]
// Note: these need to be exactly the same and in the exact same order as OpenVM's MulOpcode
// in order to be able to re-use the original Mul core chip.
// We do re-use the `BaseAluOpcode` type for 32-bit operations, but need a new opcode/enum for
// 64-bit ops.
pub enum Mul64Opcode {
    MUL,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x2254]
#[repr(usize)]
#[allow(non_camel_case_types)]
// Note: these need to be exactly the same and in the exact same order as OpenVM's DivRemOpcode
// in order to be able to re-use the original DivRem core chip.
// We do re-use the `BaseAluOpcode` type for 32-bit operations, but need a new opcode/enum for
// 64-bit ops.
pub enum DivRem64Opcode {
    DIV,
    DIVU,
    REM,
    REMU,
}

// =================================================================================================
// HintStore Instruction
// =================================================================================================

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumCount, EnumIter, FromRepr, LocalOpcode,
)]
#[opcode_offset = 0x1260]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum HintStoreOpcode {
    HINT_STOREW,
    HINT_BUFFER,
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumCount,
    EnumIter,
    FromRepr,
    LocalOpcode,
    Serialize,
    Deserialize,
)]
#[opcode_offset = 0x127A]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum ConstOpcodes {
    CONST32, // stores an immediate into a register
}

// =================================================================================================
// Phantom opcodes
// =================================================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromRepr)]
#[repr(u16)]
pub enum Phantom {
    /// Prepare the next input vector for hinting, but prepend it with a 4-byte decomposition of
    /// its length instead of one field element.
    HintInput = 0x120,
    /// Peek string from memory and print it to stdout.
    PrintStr,
    /// Prepare given amount of random numbers for hinting.
    HintRandom,
    /// Debug: trace WASI syscall execution. c_upper = syscall ID.
    TraceSyscall,
    /// Fill hint stream with 8 bytes of incrementing clock timestamp.
    ClockTimeGet,
}

// =================================================================================================
// Int256 opcodes
// =================================================================================================
//
// Newtype wrappers over the 32-bit opcode enums, exactly as OpenVM's bigint extension does:
// the 256-bit chips reuse the same core AIRs, so they reuse the same operation enums and only
// need a distinct class offset. Offsets mirror upstream's relative layout (0x400 + n) shifted
// into a free crush range; 0x1310..=0x1313 are reserved for the keccak and sha2 branches.

#[derive(Copy, Clone, Debug, LocalOpcode)]
#[opcode_offset = 0x1400]
pub struct BaseAlu256Opcode(pub BaseAluOpcode);

impl BaseAlu256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        BaseAluOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, LocalOpcode)]
#[opcode_offset = 0x1405]
pub struct Shift256Opcode(pub ShiftOpcode);

impl Shift256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        ShiftOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, LocalOpcode)]
#[opcode_offset = 0x1408]
pub struct LessThan256Opcode(pub LessThanOpcode);

impl LessThan256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        LessThanOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, LocalOpcode)]
#[opcode_offset = 0x1420]
pub struct BranchEqual256Opcode(pub BranchEqualOpcode);

impl BranchEqual256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        BranchEqualOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, LocalOpcode)]
#[opcode_offset = 0x1425]
pub struct BranchLessThan256Opcode(pub BranchLessThanOpcode);

impl BranchLessThan256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        BranchLessThanOpcode::iter().map(Self)
    }
}

#[derive(Copy, Clone, Debug, LocalOpcode)]
#[opcode_offset = 0x1450]
pub struct Mul256Opcode(pub MulOpcode);

impl Mul256Opcode {
    pub fn iter() -> impl Iterator<Item = Self> {
        MulOpcode::iter().map(Self)
    }
}
