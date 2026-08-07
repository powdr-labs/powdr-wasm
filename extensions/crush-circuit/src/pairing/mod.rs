//! Pairing support, ported from OpenVM's `openvm-pairing-circuit`.
//!
//! Unlike the other extensions this one has **no chips**: `extend_circuit` adds no AIRs
//! and its only executor variant is a phantom. It supplies hints (Fp12 inverse, line
//! evaluations, final-exponentiation witnesses) that a guest pairing implementation
//! consumes; the arithmetic itself is proved by the modular, Fp2 and ecc chips. There is
//! therefore no adapter to make frame-pointer relative and no core to reuse -- phantoms
//! read no registers.

pub use openvm_pairing_guest::{
    bls12_381::{BLS12_381_COMPLEX_STRUCT_NAME, BLS12_381_ECC_STRUCT_NAME},
    bn254::BN254_COMPLEX_STRUCT_NAME,
};

mod config;
mod fp12;
mod pairing_extension;

pub use config::*;
pub use fp12::*;
pub use pairing_extension::*;

// Unconditional: there is no GPU builder to select between. The extension contributes no
// AIRs, and `CrushGpuBuilder` rejects the arithmetic extensions a guest pairing needs.
pub use config::Rv32PairingCpuBuilder as Rv32PairingBuilder;
