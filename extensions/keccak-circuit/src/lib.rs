// The #[create_handler] macro generates unsafe fn bodies that call unsafe functions
// without inner unsafe blocks (pre-2024-edition pattern from upstream OpenVM).
#![allow(unsafe_op_in_unsafe_fn)]

//! Stateful keccak256 hasher with frame pointer support.
//! Handles full keccak sponge (padding, absorb, keccak-f) on variable length inputs
//! read from VM memory. Register addresses are offset by the frame pointer.
//!
//! Reuses sponge columns, constants, and utility functions from the upstream
//! `openvm-keccak256-circuit` crate. Only the FP-specific parts (instruction columns,
//! memory columns, register read logic) are defined here.

pub mod air;
pub mod columns;
pub(crate) mod crush_compat;
pub mod execution;
pub mod extension;
pub mod trace;

pub use air::KeccakVmAir;
pub use extension::*;

// Re-export upstream constants and utilities so consumers don't need to depend on
// openvm-keccak256-circuit directly.
pub use openvm_keccak256_circuit::{
    KECCAK_CAPACITY_BYTES, KECCAK_CAPACITY_U16S, KECCAK_DIGEST_BYTES, KECCAK_DIGEST_U64S,
    KECCAK_RATE_BYTES, KECCAK_RATE_U16S, KECCAK_WIDTH_BYTES, KECCAK_WIDTH_U16S, NUM_ABSORB_ROUNDS,
};

/// Re-export upstream utilities (keccak_f, keccak256, num_keccak_f).
pub use openvm_keccak256_circuit::utils;

use openvm_circuit::arch::*;
use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;

pub type KeccakVmChip<F> = VmChipWrapper<F, KeccakVmFiller>;

#[derive(derive_new::new, Clone, Copy)]
pub struct KeccakVmExecutor {
    pub offset: usize,
    pub pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct KeccakVmFiller {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub pointer_max_bits: usize,
}

// ==== FP-specific constants ====
/// Register reads: 1 FP read + 3 register reads (dst, src, len)
pub(crate) const KECCAK_REGISTER_READS: usize = 4;
/// Number of cells to read/write in a single memory access
pub(crate) const KECCAK_WORD_SIZE: usize = 4;
/// Memory reads for absorb per row
pub(crate) const KECCAK_ABSORB_READS: usize = KECCAK_RATE_BYTES / KECCAK_WORD_SIZE;
/// Memory writes for digest per row
pub(crate) const KECCAK_DIGEST_WRITES: usize = KECCAK_DIGEST_BYTES / KECCAK_WORD_SIZE;
