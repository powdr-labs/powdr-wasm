// Upstream is edition 2021; the #[create_handler] bodies warn under edition 2024.
#![allow(unsafe_op_in_unsafe_fn)]

//! Weierstrass elliptic-curve arithmetic, ported from OpenVM's `openvm-ecc-circuit`.
//!
//! `FieldExpressionCoreAir`/`Filler` are reused unchanged; the crush-specific parts are
//! the heap adapters (`crate::adapters`) and the two interpreters here.

use openvm_circuit::arch::DEFAULT_BLOCK_SIZE;

mod extension;
mod weierstrass_chip;

pub use extension::*;
// Re-export limb constants from algebra for consistency
pub use openvm_algebra_circuit::{NUM_LIMBS_32, NUM_LIMBS_48};
pub use weierstrass_chip::*;

// Blocks per ECC operation (2 coordinates per point)
/// Blocks for ECC with 32-limb coordinates: 2 * (32 / 4) = 16 blocks
pub const ECC_BLOCKS_32: usize = 2 * (NUM_LIMBS_32 / DEFAULT_BLOCK_SIZE);
/// Blocks for ECC with 48-limb coordinates: 2 * (48 / 4) = 24 blocks
pub const ECC_BLOCKS_48: usize = 2 * (NUM_LIMBS_48 / DEFAULT_BLOCK_SIZE);
