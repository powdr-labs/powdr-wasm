//! Chained SHA-2 using the crush SHA256 / SHA512 compression precompiles.
//!
//! Reads a variant selector, an iteration count and an expected first digest byte,
//! then hashes a digest-sized buffer that many times (each digest feeding the next)
//! and asserts the first byte matches. Chaining means the result cannot be
//! constant-folded away, and makes the whole run depend on every compression being
//! correct.
//!
//! variant 0 = SHA-256, variant 1 = SHA-512.

pub fn main() {
    let variant: u32 = crush_guest_io::read_u32();
    let n: u32 = crush_guest_io::read_u32();
    let expected: u32 = crush_guest_io::read_u32();

    let first_byte = if variant == 0 {
        let mut output = [0u8; 32];
        for _ in 0..n {
            output = crush_guest_sha2::sha256(&output);
        }
        output[0]
    } else {
        let mut output = [0u8; 64];
        for _ in 0..n {
            output = crush_guest_sha2::sha512(&output);
        }
        output[0]
    };

    assert_eq!(first_byte as u32, expected);
}
