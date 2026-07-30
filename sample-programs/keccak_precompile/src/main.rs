//! Chained keccak256 using the crush KECCAKF / XORIN precompiles.
//!
//! Reads an iteration count and an expected first digest byte, hashes a 32-byte
//! buffer that many times (each digest feeding the next), then asserts the first
//! byte matches. Chaining means the result cannot be constant-folded away, and
//! makes the whole run depend on every permutation being correct.

pub fn main() {
    let n: u32 = crush_guest_io::read_u32();
    let expected: u32 = crush_guest_io::read_u32();

    let mut output = [0u8; 32];
    for _ in 0..n {
        output = crush_guest_keccak::keccak256(&output);
    }

    assert_eq!(output[0] as u32, expected);
}
