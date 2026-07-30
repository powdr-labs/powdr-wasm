// The #[create_handler] macro generates unsafe fn bodies that call unsafe functions without
// inner unsafe blocks. Upstream's keccak crate is edition 2021, where that is allowed;
// crush-circuit is edition 2024, where it warns. Scoped to this module so the rest of the
// crate keeps the stricter default.
#![allow(unsafe_op_in_unsafe_fn)]

//! keccak256 support, ported from OpenVM's `openvm-keccak256-circuit` (v2.0.0-beta.2).
//!
//! Upstream splits keccak256 into two opcodes driven by a guest-side sponge loop:
//!
//! - `KECCAKF` applies the keccak-f permutation in place to a 200-byte state buffer
//!   ([`keccakf_op`]), delegating the permutation itself to a periphery chip over a
//!   direct bus ([`keccakf_perm`]).
//! - `XORIN` XORs a block of input bytes into the sponge buffer ([`xorin`]).
//!
//! Padding and the absorb loop live in the guest, not in any AIR.
//!
//! The only crush-specific change is that register reads are frame-pointer relative:
//! each chip reads FP from `FP_AS` first, then addresses registers at `fp + reg_ptr`.
//! `keccakf_perm` needs no change at all — it touches no registers and its whole
//! interface is the state bus — and is a verbatim copy purely because upstream
//! declares its `keccakf_perm` module privately.

pub mod keccakf_op;
pub mod keccakf_perm;
pub mod xorin;

mod constants;
pub use constants::*;

mod extension;
pub use extension::*;
