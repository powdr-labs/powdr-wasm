use std::ops::Mul;

use openvm_circuit::{
    arch::{VmStateMut, execution_mode::ExecutionCtxTrait},
    system::memory::{
        MemoryAddress,
        merkle::public_values::PUBLIC_VALUES_AS,
        online::{GuestMemory, TracingMemory},
    },
};
use openvm_instructions::{
    DEFERRAL_AS as NATIVE_AS,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
};
use openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField32};

use crate::memory_config::FP_AS;

mod alu;
pub mod call;
mod jump;
mod loadstore;

pub use alu::*;
pub use call::*;
pub use jump::*;
pub use loadstore::*;
pub use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

/// Number of 4-byte register operations to access a 32-bit value.
pub const W32_REG_OPS: usize = 1;

/// Bytes per 64-bit register pair (two 32-bit registers).
pub const W64_NUM_LIMBS: usize = 2 * RV32_REGISTER_NUM_LIMBS;

/// Number of 4-byte register operations to access a full 64-bit value.
pub const W64_REG_OPS: usize = 2;

/// 256-bit heap integer stored as 32 bytes (32 limbs of 8-bits)
pub const INT256_NUM_LIMBS: usize = 32;

// For soundness, should be <= 16
pub const RV_IS_TYPE_IMM_BITS: usize = 12;

// Branch immediate value is in [-2^12, 2^12)
pub const RV_B_TYPE_IMM_BITS: usize = 13;

pub const RV_J_TYPE_IMM_BITS: usize = 21;

#[inline(always)]
pub fn fp_addr<F: PrimeCharacteristicRing>() -> MemoryAddress<F, F> {
    MemoryAddress::new(F::from_u32(FP_AS), F::ZERO)
}

/// Build the 4-cell data block sent on the memory bus for an FP_AS access.
/// Cell 0 carries the FP value; cells 1..4 are zero — matches what
/// `FpMemory::set_fp` and the call adapter write into FP_AS, and the
/// 4-cell `DEFAULT_BLOCK_SIZE` the v2 access adapter expects.
#[inline]
pub fn fp_block<E: PrimeCharacteristicRing>(fp: E) -> [E; 4] {
    [fp, E::ZERO, E::ZERO, E::ZERO]
}

#[inline(always)]
pub fn reg_addr<F: PrimeCharacteristicRing>(ptr: F) -> MemoryAddress<F, F> {
    MemoryAddress::new(F::from_u32(RV32_REGISTER_AS), ptr)
}

#[inline(always)]
pub fn mem_addr<F: PrimeCharacteristicRing>(ptr: F) -> MemoryAddress<F, F> {
    MemoryAddress::new(F::from_u32(RV32_MEMORY_AS), ptr)
}

/// Convert the RISC-V register data (32 bits represented as 4 bytes, where each byte is represented
/// as a field element) back into its value as u32.
pub fn compose<F: PrimeField32>(ptr_data: [F; RV32_REGISTER_NUM_LIMBS]) -> u32 {
    let mut val = 0;
    for (i, limb) in ptr_data.map(|x| x.as_canonical_u32()).iter().enumerate() {
        val += limb << (i * 8);
    }
    val
}

/// inverse of `compose`
pub fn decompose<F: PrimeField32>(value: u32) -> [F; RV32_REGISTER_NUM_LIMBS] {
    std::array::from_fn(|i| {
        F::from_u32((value >> (RV32_CELL_BITS * i)) & ((1 << RV32_CELL_BITS) - 1))
    })
}

/// Convert a 24-bit encoded immediate to `[u8; N]` with sign extension.
/// The immediate is stored as 3 bytes: low, mid, sign. Byte 2 (the sign byte)
/// is replicated to all higher positions. Works correctly for negative numbers:
/// e.g., -5 is encoded as 0x00FFFB → [0xFB, 0xFF, 0xFF, 0xFF] for N=4.
#[inline(always)]
pub fn imm_to_bytes<const N: usize>(imm: u32) -> [u8; N] {
    debug_assert_eq!(imm >> 24, 0);
    let imm_le = imm.to_le_bytes();
    let sign_byte = imm_le[2];
    std::array::from_fn(|i| if i < 3 { imm_le[i] } else { sign_byte })
}

#[inline(always)]
pub fn memory_read<const N: usize>(memory: &GuestMemory, address_space: u32, ptr: u32) -> [u8; N] {
    debug_assert!(
        address_space == RV32_REGISTER_AS
            || address_space == RV32_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS,
    );

    // SAFETY:
    // - address space `RV32_REGISTER_AS` and `RV32_MEMORY_AS` will always have cell type `u8` and
    //   minimum alignment of `RV32_REGISTER_NUM_LIMBS`
    unsafe { memory.read::<u8, N>(address_space, ptr) }
}

#[inline(always)]
pub fn memory_write<const N: usize>(
    memory: &mut GuestMemory,
    address_space: u32,
    ptr: u32,
    data: [u8; N],
) {
    debug_assert!(
        address_space == RV32_REGISTER_AS
            || address_space == RV32_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS
    );

    // SAFETY:
    // - address space `RV32_REGISTER_AS` and `RV32_MEMORY_AS` will always have cell type `u8` and
    //   minimum alignment of `RV32_REGISTER_NUM_LIMBS`
    unsafe { memory.write::<u8, N>(address_space, ptr, data) }
}

/// Read from address space `NATIVE_AS` (= `DEFERRAL_AS`), which has cell type `F`.
/// Crush-specific: upstream rv32im rejects `DEFERRAL_AS` in loadstore; we need it
/// to support stores into the deferral address space.
#[inline(always)]
pub fn memory_read_native<F, const N: usize>(memory: &GuestMemory, ptr: u32) -> [F; N]
where
    F: PrimeField32,
{
    // SAFETY: address space `NATIVE_AS` always has cell type `F` and minimum alignment `1`.
    unsafe { memory.read::<F, N>(NATIVE_AS, ptr) }
}

/// Atomic write into address space `NATIVE_AS` (= `DEFERRAL_AS`), which has cell type `F`.
/// Crush-specific: upstream rv32im rejects `DEFERRAL_AS` in loadstore; we need it
/// to support stores into the deferral address space.
#[inline(always)]
pub fn timed_write_native<F, const N: usize>(
    memory: &mut TracingMemory,
    ptr: u32,
    vals: [F; N],
) -> (u32, [F; N])
where
    F: PrimeField32,
{
    // SAFETY: address space `NATIVE_AS` always has cell type `F` and minimum alignment `1`.
    unsafe { memory.write::<F, N>(NATIVE_AS, ptr, vals) }
}

/// Atomic read operation which increments the timestamp by 1.
/// Returns `(t_prev, [ptr:4]_{address_space})` where `t_prev` is the timestamp of the last memory
/// access.
#[inline(always)]
pub fn timed_read<const N: usize>(
    memory: &mut TracingMemory,
    address_space: u32,
    ptr: u32,
) -> (u32, [u8; N]) {
    debug_assert!(
        address_space == RV32_REGISTER_AS
            || address_space == RV32_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS
    );

    // SAFETY:
    // - address space `RV32_REGISTER_AS` and `RV32_MEMORY_AS` will always have cell type `u8` and
    //   minimum alignment of `RV32_REGISTER_NUM_LIMBS`
    unsafe { memory.read::<u8, N>(address_space, ptr) }
}

#[inline(always)]
pub fn timed_write<const N: usize>(
    memory: &mut TracingMemory,
    address_space: u32,
    ptr: u32,
    data: [u8; N],
) -> (u32, [u8; N]) {
    debug_assert!(
        address_space == RV32_REGISTER_AS
            || address_space == RV32_MEMORY_AS
            || address_space == PUBLIC_VALUES_AS
    );

    // SAFETY:
    // - address space `RV32_REGISTER_AS` and `RV32_MEMORY_AS` will always have cell type `u8` and
    //   minimum alignment of `RV32_REGISTER_NUM_LIMBS`
    unsafe { memory.write::<u8, N>(address_space, ptr, data) }
}

/// Reads register value at `reg_ptr` from memory and records the memory access in mutable buffer.
/// Trace generation relevant to this memory access can be done fully from the recorded buffer.
#[inline(always)]
pub fn tracing_read<const N: usize>(
    memory: &mut TracingMemory,
    address_space: u32,
    ptr: u32,
    prev_timestamp: &mut u32,
) -> [u8; N] {
    let (t_prev, data) = timed_read(memory, address_space, ptr);
    *prev_timestamp = t_prev;
    data
}

/// Reads an immediate value, increments the timestamp, and returns sign-extended bytes.
/// Records the immediate value in the mutable record buffer.
#[inline(always)]
pub fn tracing_read_imm<const N: usize>(
    memory: &mut TracingMemory,
    imm: u32,
    imm_mut: &mut u32,
) -> [u8; N] {
    *imm_mut = imm;
    debug_assert_eq!(imm >> 24, 0);

    memory.increment_timestamp();

    imm_to_bytes(imm)
}

/// Writes `reg_ptr, reg_val` into memory and records the memory access in mutable buffer.
/// Trace generation relevant to this memory access can be done fully from the recorded buffer.
#[inline(always)]
pub fn tracing_write<const N: usize>(
    memory: &mut TracingMemory,
    address_space: u32,
    ptr: u32,
    data: [u8; N],
    prev_timestamp: &mut u32,
    prev_data: &mut [u8; N],
) {
    let (t_prev, data_prev) = timed_write(memory, address_space, ptr, data);
    *prev_timestamp = t_prev;
    *prev_data = data_prev;
}

#[inline(always)]
pub fn memory_read_from_state<F, Ctx, const N: usize>(
    state: &mut VmStateMut<F, GuestMemory, Ctx>,
    address_space: u32,
    ptr: u32,
) -> [u8; N]
where
    Ctx: ExecutionCtxTrait,
{
    state.ctx.on_memory_operation(address_space, ptr, N as u32);

    memory_read(state.memory, address_space, ptr)
}

#[inline(always)]
pub fn memory_write_from_state<F, Ctx, const N: usize>(
    state: &mut VmStateMut<F, GuestMemory, Ctx>,
    address_space: u32,
    ptr: u32,
    data: [u8; N],
) where
    Ctx: ExecutionCtxTrait,
{
    state.ctx.on_memory_operation(address_space, ptr, N as u32);

    memory_write(state.memory, address_space, ptr, data)
}

#[inline(always)]
pub fn read_rv32_register_from_state<F, Ctx>(
    state: &mut VmStateMut<F, GuestMemory, Ctx>,
    ptr: u32,
) -> u32
where
    Ctx: ExecutionCtxTrait,
{
    u32::from_le_bytes(memory_read_from_state(state, RV32_REGISTER_AS, ptr))
}

#[inline(always)]
pub fn read_rv32_register(memory: &GuestMemory, ptr: u32) -> u32 {
    u32::from_le_bytes(memory_read(memory, RV32_REGISTER_AS, ptr))
}

pub fn abstract_compose<T: PrimeCharacteristicRing, V: Mul<T, Output = T>>(
    data: [V; RV32_REGISTER_NUM_LIMBS],
) -> T {
    data.into_iter()
        .enumerate()
        .fold(T::ZERO, |acc, (i, limb)| {
            acc + limb * T::from_u32(1 << (i * RV32_CELL_BITS))
        })
}

/// Tracing read of the frame pointer from FP_AS address 0.
/// Returns the fp value and records the previous timestamp for trace generation.
#[inline(always)]
pub fn tracing_read_fp<F: PrimeField32>(
    memory: &mut TracingMemory,
    prev_timestamp: &mut u32,
) -> u32 {
    // SAFETY: FP_AS uses native32 cell type (F). Block size 4 matches v2's
    // DEFAULT_BLOCK_SIZE; only cell 0 carries the FP value.
    let (t_prev, data) = unsafe { memory.read::<F, 4>(FP_AS, 0) };
    *prev_timestamp = t_prev;
    data[0].as_canonical_u32()
}

// TEMP[jpw]
pub fn tmp_convert_to_u8s<F: PrimeField32, const N: usize>(data: [F; N]) -> [u8; N] {
    data.map(|x| x.as_canonical_u32() as u8)
}
