# powdr-wasm ISA Reference

This document describes the instruction set implemented by powdr-wasm. Each instruction is documented with its encoding, semantics, and any relevant preconditions.

powdr-wasm reuses common parts of [OpenVM](https://github.com/openvm-org/openvm/)'s RISC-V implementation. Many of the architecture choices, constants, address spaces, and chip implementations are taken from OpenVM's RV32IM extension.

See `extensions/openvm-transpiler/src/instructions.rs` for opcode definitions and `extensions/crush-translation/src/instruction_builder.rs` for encoding helpers.

## Execution Model

Registers are **frame-pointer-relative**: every register access uses `FP + offset` as the address in the register address space (`RV32_REGISTER_AS = 1`). The current frame pointer is stored at address 0 in a dedicated address space (`FP_AS = 5`).

All 32-bit values are stored as 4 little-endian bytes (RV32_REGISTER_NUM_LIMBS = 4). 64-bit values use 8 bytes.

**64-bit register layout:** 64-bit instructions still take two register indices (rs1, rs2) in the encoding. The 64-bit value is composed of two consecutive 32-bit registers: `[rs1]` holds the low 32-bit limb, and `[rs1+1]` holds the high 32-bit limb. Same for rs2 and rd. Not all 64-bit instructions produce 64-bit output — comparison and equality operations output only 32 bits (1 or 0) to rd.

The program counter advances by `DEFAULT_PC_STEP` (4) after each instruction unless the instruction modifies it.

### Address Spaces

Address spaces 1, 2, and 3 are the same as in OpenVM's RISC-V implementation. Address space 5 is added by powdr-wasm for frame pointer storage.

| AS | Constant | Purpose |
|----|----------|---------|
| 1 | `RV32_REGISTER_AS` | Register file (4-byte-aligned u8 cells) |
| 2 | `RV32_MEMORY_AS` | Main memory (WebAssembly linear memory, tables, globals, initialization constants, etc.) |
| 3 | `PUBLIC_VALUES_AS` | Public outputs |
| 5 | `FP_AS` | Frame pointer storage (1 native field element at address 0) |

### Instruction Encoding

Instructions have 8 fields: `opcode, a, b, c, d, e, f, g`. The encoding varies by instruction format.

#### R-type (register-register-register)

Used by ALU, shift, comparison, and equality instructions.

| Field | Value |
|-------|-------|
| a | `RV32_REGISTER_NUM_LIMBS * rd` |
| b | `RV32_REGISTER_NUM_LIMBS * rs1` |
| c | `RV32_REGISTER_NUM_LIMBS * rs2` |
| d | 1 (register AS) |
| e | 1 (register AS) |
| f | 0 |
| g | 0 |

#### I-type (register-register-immediate)

Used by ALU/shift/comparison instructions with an immediate operand.

| Field | Value |
|-------|-------|
| a | `RV32_REGISTER_NUM_LIMBS * rd` |
| b | `RV32_REGISTER_NUM_LIMBS * rs1` |
| c | `AluImm`-encoded 16-bit signed immediate: `i16` sign-extended to 24 bits (`value as i32 as u32 & 0xff_ff_ff`) |
| d | 1 (register AS) |
| e | 0 (signals immediate mode) |
| f | 0 |
| g | 0 |

---

## Arithmetic Instructions (32-bit)

Opcodes from `BaseAluOpcode` (re-exported from OpenVM RV32IM), offset `0x100`.

| Instruction | Opcode | Format | Semantics |
|-------------|--------|--------|-----------|
| `add` | `BaseAluOpcode::ADD` | R / I | `rd = rs1 + rs2` (wrapping) |
| `sub` | `BaseAluOpcode::SUB` | R / I | `rd = rs1 - rs2` (wrapping) |
| `xor` | `BaseAluOpcode::XOR` | R / I | `rd = rs1 ^ rs2` |
| `or` | `BaseAluOpcode::OR` | R / I | `rd = rs1 \| rs2` |
| `and` | `BaseAluOpcode::AND` | R / I | `rd = rs1 & rs2` |

## Arithmetic Instructions (64-bit)

Opcodes from `BaseAlu64Opcode`, offset `0x2200`. Same variant names and order as `BaseAluOpcode` to reuse the same ALU core chip.

| Instruction | Opcode | Format | Semantics |
|-------------|--------|--------|-----------|
| `add_64` | `BaseAlu64Opcode::ADD` | R / I | `rd = rs1 + rs2` (wrapping, 64-bit) |
| `sub_64` | `BaseAlu64Opcode::SUB` | R / I | `rd = rs1 - rs2` (wrapping, 64-bit) |
| `xor_64` | `BaseAlu64Opcode::XOR` | R / I | `rd = rs1 ^ rs2` (64-bit) |
| `or_64` | `BaseAlu64Opcode::OR` | R / I | `rd = rs1 \| rs2` (64-bit) |
| `and_64` | `BaseAlu64Opcode::AND` | R / I | `rd = rs1 & rs2` (64-bit) |

## Multiplication

32-bit opcodes from `MulOpcode` (re-exported from OpenVM RV32IM), offset `0x0100`. 64-bit opcodes from `Mul64Opcode`, offset `0x2250`.

| Instruction | Opcode | Format | Semantics |
|-------------|--------|--------|-----------|
| `mul` | `MulOpcode::MUL` | R / I | `rd = (rs1 * rs2)[31:0]` (low 32 bits) |
| `mul_64` | `Mul64Opcode::MUL` | R / I | `rd = (rs1 * rs2)[63:0]` (low 64 bits) |

## Division and Remainder

32-bit opcodes from `DivRemOpcode` (re-exported from OpenVM RV32IM), offset `0x0100`. 64-bit opcodes from `DivRem64Opcode`, offset `0x2254`.

**Division by zero:** The circuit constraints follow the RISC-V specification for division by zero, which differs from WebAssembly semantics. See [#24](https://github.com/powdr-labs/powdr-wasm/issues/24) for details.

| Instruction | Opcode | Format | Semantics |
|-------------|--------|--------|-----------|
| `div` | `DivRemOpcode::DIV` | R | `rd = rs1 /s rs2` (signed) |
| `divu` | `DivRemOpcode::DIVU` | R | `rd = rs1 /u rs2` (unsigned) |
| `rem` | `DivRemOpcode::REM` | R | `rd = rs1 %s rs2` (signed) |
| `remu` | `DivRemOpcode::REMU` | R | `rd = rs1 %u rs2` (unsigned) |
| `div_64` | `DivRem64Opcode::DIV` | R | `rd = rs1 /s rs2` (signed, 64-bit) |
| `divu_64` | `DivRem64Opcode::DIVU` | R | `rd = rs1 /u rs2` (unsigned, 64-bit) |
| `rem_64` | `DivRem64Opcode::REM` | R | `rd = rs1 %s rs2` (signed, 64-bit) |
| `remu_64` | `DivRem64Opcode::REMU` | R | `rd = rs1 %u rs2` (unsigned, 64-bit) |

## Shift Instructions

32-bit opcodes from `ShiftOpcode` (re-exported from OpenVM RV32IM), offset `0x0100`. 64-bit opcodes from `Shift64Opcode`, offset `0x2205`.

| Instruction | Opcode | Format | Semantics |
|-------------|--------|--------|-----------|
| `shl` | `ShiftOpcode::SLL` | R / I | `rd = rs1 << rs2` |
| `shr_u` | `ShiftOpcode::SRL` | R / I | `rd = rs1 >>u rs2` (logical) |
| `shr_s` | `ShiftOpcode::SRA` | R / I | `rd = rs1 >>s rs2` (arithmetic) |
| `shl_64` | `Shift64Opcode::SLL` | R / I | `rd = rs1 << rs2` (64-bit) |
| `shr_u_64` | `Shift64Opcode::SRL` | R / I | `rd = rs1 >>u rs2` (64-bit, logical) |
| `shr_s_64` | `Shift64Opcode::SRA` | R / I | `rd = rs1 >>s rs2` (64-bit, arithmetic) |

## Comparison Instructions

32-bit opcodes from `LessThanOpcode` (re-exported from OpenVM RV32IM), offset `0x0100`. 64-bit opcodes from `LessThan64Opcode`, offset `0x2208`.

The 64-bit comparison operations take 64-bit inputs but always output a 32-bit result (1 or 0) to rd.

| Instruction | Opcode | Format | Semantics |
|-------------|--------|--------|-----------|
| `lt_s` | `LessThanOpcode::SLT` | R | `rd = (rs1 <s rs2) ? 1 : 0` (signed) |
| `lt_u` | `LessThanOpcode::SLTU` | R / I | `rd = (rs1 <u rs2) ? 1 : 0` (unsigned) |
| `gt_s` | (SLT swapped) | R | `rd = (rs1 >s rs2) ? 1 : 0` — emitted as `lt_s(rd, rs2, rs1)` |
| `gt_u` | (SLTU swapped) | R | `rd = (rs1 >u rs2) ? 1 : 0` — emitted as `lt_u(rd, rs2, rs1)` |
| `lt_s_64` | `LessThan64Opcode::SLT` | R | `rd = (rs1 <s rs2) ? 1 : 0` (signed, 64-bit inputs, 32-bit output) |
| `lt_u_64` | `LessThan64Opcode::SLTU` | R | `rd = (rs1 <u rs2) ? 1 : 0` (unsigned, 64-bit inputs, 32-bit output) |
| `gt_s_64` | (SLT swapped) | R | `rd = (rs1 >s rs2) ? 1 : 0` — emitted as `lt_s_64(rd, rs2, rs1)` |
| `gt_u_64` | (SLTU swapped) | R | `rd = (rs1 >u rs2) ? 1 : 0` — emitted as `lt_u_64(rd, rs2, rs1)` |

`ge_s`/`ge_u`/`le_s`/`le_u` and `eqz` are synthesized by the crush translator using the primitives above (e.g., `ge_u` = inverted `lt_u`, `eqz` = `lt_u` with immediate 1).

## Equality Instructions

32-bit opcodes from `EqOpcode`, offset `0x120c`. 64-bit opcodes from `Eq64Opcode`, offset `0x220c`.

The 64-bit equality operations take 64-bit inputs but always output a 32-bit result (1 or 0) to rd.

| Instruction | Opcode | Format | Semantics |
|-------------|--------|--------|-----------|
| `eq` | `EqOpcode::EQ` | R / I | `rd = (rs1 == rs2) ? 1 : 0` |
| `neq` | `EqOpcode::NEQ` | R / I | `rd = (rs1 != rs2) ? 1 : 0` |
| `eq_64` | `Eq64Opcode::EQ` | R / I | `rd = (rs1 == rs2) ? 1 : 0` (64-bit inputs, 32-bit output) |
| `neq_64` | `Eq64Opcode::NEQ` | R / I | `rd = (rs1 != rs2) ? 1 : 0` (64-bit inputs, 32-bit output) |

---

## Constant Loading

Opcode from `ConstOpcodes`, offset `0x127A`.

### CONST32

Loads a 32-bit immediate into a register.

| Field | Value |
|-------|-------|
| a | `RV32_REGISTER_NUM_LIMBS * target_reg` |
| b | `imm_lo` (lower 16 bits of immediate) |
| c | `imm_hi` (upper 16 bits of immediate) |

**Semantics:** Writes `(imm_hi << 16) | imm_lo` as 4 little-endian bytes to the target register.

**Note:** The immediate is split into two 16-bit halves due to the instruction encoding's field size. If the instruction were extended to use 4 bytes for the immediate, the ZK constraints could be simplified.

---

## Control Flow

### Call / Return

Opcodes from `CallOpcode`, offset `0x1236`.

#### RET

Return to a saved PC and restore the frame pointer.

**Precondition:** `RET` will only receive values that were previously written by a `CALL` or `CALL_INDIRECT`. The target PC and FP registers must contain valid saved values.

| Field | Value |
|-------|-------|
| a | 0 |
| b | 0 |
| c | `RV32_REGISTER_NUM_LIMBS * to_pc_reg` (register offset holding saved PC, relative to current FP) |
| d | `RV32_REGISTER_NUM_LIMBS * to_fp_reg` (register offset holding saved FP, relative to current FP) |
| e | 1 (PC read from register AS) |
| f | 1 (FP read from register AS) |

**Semantics:**
1. Read current FP from FP_AS.
2. Read new PC from register `[FP + c]`.
3. Read new FP (absolute value) from register `[FP + d]`.
4. Update FP_AS with new FP.
5. Set PC = new PC.

#### CALL

Call a function at an immediate PC address.

| Field | Value |
|-------|-------|
| a | `RV32_REGISTER_NUM_LIMBS * save_pc` (where to save return PC (current PC + 4), relative to **new** FP) |
| b | `RV32_REGISTER_NUM_LIMBS * save_fp` (where to save the current FP, relative to **new** FP) |
| c | `to_pc_imm` (immediate target PC) |
| d | `fp_offset` (added to current FP to compute new FP) |
| e | 0 (PC from immediate) |
| f | 0 (FP from immediate offset) |

**Semantics:**
1. Read current FP from FP_AS.
2. Compute `new_fp = current_fp + d`.
3. Save `current_pc + DEFAULT_PC_STEP` to register `[new_fp + a]`.
4. Save `current_fp` to register `[new_fp + b]`.
5. Update FP_AS with `new_fp`.
6. Set PC = c.

#### CALL_INDIRECT

Call a function at a PC address read from a register.

| Field | Value |
|-------|-------|
| a | `RV32_REGISTER_NUM_LIMBS * save_pc` (where to save return PC (current PC + 4), relative to **new** FP) |
| b | `RV32_REGISTER_NUM_LIMBS * save_fp` (where to save the current FP, relative to **new** FP) |
| c | `RV32_REGISTER_NUM_LIMBS * to_pc_reg` (register holding target PC, relative to current FP) |
| d | `fp_offset` (added to current FP to compute new FP) |
| e | 1 (PC from register) |
| f | 0 (FP from immediate offset) |

**Semantics:** Same as `CALL`, except the target PC is read from register `[FP + c]` instead of being an immediate.

### Jump

Opcodes from `JumpOpcode`, offset `0x123B`.

#### JUMP

Unconditional jump to an immediate PC.

| Field | Value |
|-------|-------|
| a | `to_pc_imm` |

**Semantics:** Set PC = a.

#### SKIP

Unconditional relative jump by a register-specified offset.

| Field | Value |
|-------|-------|
| b | `RV32_REGISTER_NUM_LIMBS * offset_reg` |

**Semantics:** Read offset from register `[FP + b]`. Set `PC += (offset + 1) * DEFAULT_PC_STEP`. The `+1` accounts for crush's natural PC increment — without it, offset 0 would loop forever.

#### JUMP_IF

Conditional jump to an immediate PC if condition register is non-zero.

| Field | Value |
|-------|-------|
| a | `to_pc_imm` |
| b | `RV32_REGISTER_NUM_LIMBS * condition_reg` |

**Semantics:** Read condition from register `[FP + b]`. If non-zero, set PC = a. Otherwise, PC += DEFAULT_PC_STEP.

#### JUMP_IF_ZERO

Conditional jump to an immediate PC if condition register is zero.

| Field | Value |
|-------|-------|
| a | `to_pc_imm` |
| b | `RV32_REGISTER_NUM_LIMBS * condition_reg` |

**Semantics:** Read condition from register `[FP + b]`. If zero, set PC = a. Otherwise, PC += DEFAULT_PC_STEP.

---

## Memory Instructions

Opcodes from `Rv32LoadStoreOpcode` (re-exported from OpenVM RV32IM as `LoadStoreOpcode`). Copied from OpenVM to keep the same behavior.

All load/store instructions use a base register + immediate offset addressing mode. The immediate is split across fields c (lower 16 bits) and g (upper 16 bits).

### Encoding (common to all load/store)

| Field | Value |
|-------|-------|
| a | `RV32_REGISTER_NUM_LIMBS * rd` (for loads) or `RV32_REGISTER_NUM_LIMBS * rs2` (for stores, the data register) |
| b | `RV32_REGISTER_NUM_LIMBS * rs1` (base address register) |
| c | `imm & 0xFFFF` (lower 16 bits of offset) |
| d | 1 (register AS) |
| e | 2 (memory AS) |
| g | `imm >> 16` (upper 16 bits of offset) |

### Load Instructions

| Instruction | Opcode | Semantics |
|-------------|--------|-----------|
| `loadw` | `LOADW` | `rd = MEM32[rs1 + imm]` — load 32-bit word |
| `loadb` | `LOADB` | `rd = sign_extend(MEM8[rs1 + imm])` — load byte, sign-extend to 32 bits |
| `loadbu` | `LOADBU` | `rd = zero_extend(MEM8[rs1 + imm])` — load byte, zero-extend to 32 bits |
| `loadh` | `LOADH` | `rd = sign_extend(MEM16[rs1 + imm])` — load halfword, sign-extend to 32 bits |
| `loadhu` | `LOADHU` | `rd = zero_extend(MEM16[rs1 + imm])` — load halfword, zero-extend to 32 bits |

### Store Instructions

| Instruction | Opcode | Semantics |
|-------------|--------|-----------|
| `storew` | `STOREW` | `MEM32[rs1 + imm] = rs2` — store 32-bit word |
| `storeb` | `STOREB` | `MEM8[rs1 + imm] = rs2[7:0]` — store lowest byte |
| `storeh` | `STOREH` | `MEM16[rs1 + imm] = rs2[15:0]` — store lowest halfword |

### Reveal (Public Output)

`reveal` uses `STOREW` with memory AS = 3 (`PUBLIC_VALUES_AS`) to write a register value into the public output area. Copied from OpenVM to keep the same behavior.

| Field | Value |
|-------|-------|
| a | `RV32_REGISTER_NUM_LIMBS * data_reg` |
| b | `RV32_REGISTER_NUM_LIMBS * index_reg` |
| c | 0 (or immediate index) |
| d | 1 (register AS) |
| e | 3 (public values AS) |

---

## Hint Stream Instructions

### Phantom Instructions

Phantom instructions modify the hint stream without producing circuit trace rows. Opcodes are encoded as discriminants in field c of a `PHANTOM` system opcode. Copied from OpenVM and modified to take the FP into account when reading registers.

| Instruction | Phantom Discriminant | Semantics |
|-------------|---------------------|-----------|
| `prepare_read` (HintInput) | `0x120` | Pop next input vector, prepend its 4-byte LE length, push onto hint stream |
| `debug_print` (PrintStr) | `0x121` | Read string from memory and print to stdout (debug only) |
| HintRandom | `0x122` | Generate random bytes and push onto hint stream |

### HintStore Instructions

Opcodes from `HintStoreOpcode`, offset `0x1260`. These consume data from the hint stream and write it to memory. Copied from OpenVM and modified to take the FP into account when reading registers.

#### HINT_STOREW

Read one word (4 bytes) from the hint stream and write to memory.

| Field | Value |
|-------|-------|
| a | 0 |
| b | `RV32_REGISTER_NUM_LIMBS * mem_ptr_reg` (register holding target memory address) |
| c | 0 |
| d | `RV32_REGISTER_AS` |
| e | `RV32_MEMORY_AS` |

**Semantics:** Pop 4 bytes from hint stream. Read memory pointer from register `[FP + b]`. Write 4 bytes to `MEM[mem_ptr]`.

#### HINT_BUFFER

Read multiple words from the hint stream and write to consecutive memory addresses.

| Field | Value |
|-------|-------|
| a | `RV32_REGISTER_NUM_LIMBS * num_words_reg` (register holding word count) |
| b | `RV32_REGISTER_NUM_LIMBS * mem_ptr_reg` (register holding target memory address) |
| c | 0 |
| d | `RV32_REGISTER_AS` |
| e | `RV32_MEMORY_AS` |

**Precondition:** `num_words > 0` (debug-asserted). The hint stream must contain at least `4 * num_words` bytes; otherwise an `ExecutionError::HintOutOfBounds` is raised.

**Semantics:** Read `num_words` from register `[FP + a]`. Read `mem_ptr` from register `[FP + b]`. For each word `i` in `0..num_words`, pop 4 bytes from hint stream and write to `MEM[mem_ptr + 4*i]`.

---

## System Instructions

These use OpenVM's built-in `SystemOpcode` rather than crush-specific opcodes.

### TERMINATE (halt / trap / abort)

| Instruction | Exit Code | Semantics |
|-------------|-----------|-----------|
| `halt` | 0 | Normal program termination |
| `trap(code)` | `100 + code` | WebAssembly trap (e.g., unreachable, out-of-bounds). Code is the crush trap reason. |
| `abort` | 200 | Explicit abort |
| (unimplemented) | 201 | Emitted for unimplemented instructions (e.g., SIMD, float). Only triggers if actually executed. |

---

## Opcode Map Summary

| Enum | Offset | Opcodes |
|------|--------|---------|
| `BaseAluOpcode` | 0x0100 | ADD, SUB, XOR, OR, AND |
| `MulOpcode` | 0x0100 | MUL |
| `DivRemOpcode` | 0x0100 | DIV, DIVU, REM, REMU |
| `LessThanOpcode` | 0x0100 | SLT, SLTU |
| `ShiftOpcode` | 0x0100 | SLL, SRL, SRA |
| `LoadStoreOpcode` | 0x0100 | LOADW, STOREW, LOADB, LOADBU, LOADH, LOADHU, STOREB, STOREH |
| `EqOpcode` | 0x120c | EQ, NEQ |
| `CallOpcode` | 0x1236 | RET, CALL, CALL_INDIRECT |
| `JumpOpcode` | 0x123B | JUMP, SKIP, JUMP_IF, JUMP_IF_ZERO |
| `HintStoreOpcode` | 0x1260 | HINT_STOREW, HINT_BUFFER |
| `ConstOpcodes` | 0x127A | CONST32 |
| `BaseAlu64Opcode` | 0x2200 | ADD, SUB, XOR, OR, AND |
| `Shift64Opcode` | 0x2205 | SLL, SRL, SRA |
| `LessThan64Opcode` | 0x2208 | SLT, SLTU |
| `Eq64Opcode` | 0x220c | EQ, NEQ |
| `Mul64Opcode` | 0x2250 | MUL |
| `DivRem64Opcode` | 0x2254 | DIV, DIVU, REM, REMU |

The 32-bit ALU/Mul/DivRem/Shift/LessThan opcodes are re-exported from OpenVM's RV32IM transpiler and share the same offset range. The 64-bit variants are crush-specific and use offset range `0x22xx` with the same variant names and order to reuse the same core chips.
