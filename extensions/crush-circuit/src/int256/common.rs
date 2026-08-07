use openvm_circuit::{
    arch::{DEFAULT_BLOCK_SIZE, ExecutionCtxTrait, VmExecState},
    system::memory::online::GuestMemory,
};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::int256::{INT256_NUM_BLOCKS, INT256_NUM_LIMBS, RV32_CELL_BITS};

/// Read a 256-bit integer as 8 separate 4-byte block reads.
#[inline(always)]
pub fn read_int256<F: PrimeField32, CTX: ExecutionCtxTrait>(
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
    addr_space: u32,
    ptr: u32,
) -> [u8; INT256_NUM_LIMBS] {
    let mut result = [0u8; INT256_NUM_LIMBS];
    for i in 0..INT256_NUM_BLOCKS {
        let block: [u8; DEFAULT_BLOCK_SIZE] =
            exec_state.vm_read(addr_space, ptr + (i * DEFAULT_BLOCK_SIZE) as u32);
        result[i * DEFAULT_BLOCK_SIZE..(i + 1) * DEFAULT_BLOCK_SIZE].copy_from_slice(&block);
    }
    result
}

/// Write a 256-bit integer as 8 separate 4-byte block writes.
#[inline(always)]
pub fn write_int256<F: PrimeField32, CTX: ExecutionCtxTrait>(
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
    addr_space: u32,
    ptr: u32,
    data: &[u8; INT256_NUM_LIMBS],
) {
    for i in 0..INT256_NUM_BLOCKS {
        let block: [u8; DEFAULT_BLOCK_SIZE] = data
            [i * DEFAULT_BLOCK_SIZE..(i + 1) * DEFAULT_BLOCK_SIZE]
            .try_into()
            .unwrap();
        exec_state.vm_write(addr_space, ptr + (i * DEFAULT_BLOCK_SIZE) as u32, &block);
    }
}

#[inline(always)]
pub fn bytes_to_u64_array(bytes: [u8; INT256_NUM_LIMBS]) -> [u64; 4] {
    // SAFETY: [u8; 32] to [u64; 4] transmute is safe - same size and compatible alignment
    unsafe { std::mem::transmute(bytes) }
}

#[inline(always)]
pub fn u64_array_to_bytes(u64_array: [u64; 4]) -> [u8; INT256_NUM_LIMBS] {
    // SAFETY: [u64; 4] to [u8; 32] transmute is safe - same size and compatible alignment
    unsafe { std::mem::transmute(u64_array) }
}

#[inline(always)]
pub fn bytes_to_u32_array(bytes: [u8; INT256_NUM_LIMBS]) -> [u32; 8] {
    // SAFETY: [u8; 32] to [u32; 8] transmute is safe - same size and compatible alignment
    unsafe { std::mem::transmute(bytes) }
}

#[inline(always)]
pub fn u32_array_to_bytes(u32_array: [u32; 8]) -> [u8; INT256_NUM_LIMBS] {
    // SAFETY: [u32; 8] to [u8; 32] transmute is safe - same size and compatible alignment
    unsafe { std::mem::transmute(u32_array) }
}

#[inline(always)]
pub(crate) fn u256_lt(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
    let rs1_u64: [u64; 4] = bytes_to_u64_array(rs1);
    let rs2_u64: [u64; 4] = bytes_to_u64_array(rs2);
    for i in (0..4).rev() {
        if rs1_u64[i] != rs2_u64[i] {
            return rs1_u64[i] < rs2_u64[i];
        }
    }
    false
}

#[inline(always)]
pub(crate) fn i256_lt(rs1: [u8; INT256_NUM_LIMBS], rs2: [u8; INT256_NUM_LIMBS]) -> bool {
    // true for negative. false for positive
    let rs1_sign = rs1[INT256_NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1) == 1;
    let rs2_sign = rs2[INT256_NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1) == 1;
    let rs1_u64: [u64; 4] = bytes_to_u64_array(rs1);
    let rs2_u64: [u64; 4] = bytes_to_u64_array(rs2);
    for i in (0..4).rev() {
        if rs1_u64[i] != rs2_u64[i] {
            return (rs1_u64[i] < rs2_u64[i]) ^ rs1_sign ^ rs2_sign;
        }
    }
    false
}
