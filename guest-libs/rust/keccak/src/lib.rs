//! keccak256 for crush guests, built on the `KECCAKF` and `XORIN` precompiles.
//!
//! The sponge itself — padding and the absorb loop — is ordinary guest code; only
//! the keccak-f permutation and the block XOR are precompiled. This mirrors how
//! OpenVM's `openvm-keccak256-guest` drives its own two opcodes.
//!
//! Input is staged in a rate-sized, 4-byte-aligned buffer and handed to `XORIN` one
//! full block at a time. That keeps every precompile call on the shape the circuit
//! wants (both pointers 4-byte aligned, length a multiple of 4 and at most the rate)
//! without the caller needing to care how its input is aligned or sized.

#![no_std]

/// Width of the keccak state in bytes (1600 bits).
pub const KECCAK_WIDTH_BYTES: usize = 200;
/// Number of rate bytes absorbed per permutation.
pub const KECCAK_RATE: usize = 136;
/// Size of a keccak256 digest in bytes.
pub const KECCAK_OUTPUT_SIZE: usize = 32;

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// XOR `len` bytes from `input` into `buffer`.
    ///
    /// Both pointers must be 4-byte aligned, and `len` must be a multiple of 4 and
    /// at most [`KECCAK_RATE`].
    unsafe fn __native_xorin(buffer: *mut u8, input: *const u8, len: usize);

    /// Apply the keccak-f[1600] permutation in place to the 200-byte state at
    /// `buffer`, which must be 4-byte aligned.
    unsafe fn __native_keccakf(buffer: *mut u8);
}

/// Software stand-ins for the precompiles, so the sponge below can be exercised off
/// target. The sponge is plain guest code rather than constrained circuitry, so its
/// padding and absorb logic is worth testing directly; see the tests at the bottom.
#[cfg(not(target_arch = "wasm32"))]
mod host {
    use super::KECCAK_WIDTH_BYTES;

    pub(super) unsafe fn __native_xorin(buffer: *mut u8, input: *const u8, len: usize) {
        for i in 0..len {
            unsafe { *buffer.add(i) ^= *input.add(i) };
        }
    }

    pub(super) unsafe fn __native_keccakf(buffer: *mut u8) {
        let mut lanes = [0u64; KECCAK_WIDTH_BYTES / 8];
        for (i, lane) in lanes.iter_mut().enumerate() {
            let mut bytes = [0u8; 8];
            for (j, b) in bytes.iter_mut().enumerate() {
                *b = unsafe { *buffer.add(i * 8 + j) };
            }
            *lane = u64::from_le_bytes(bytes);
        }
        tiny_keccak::keccakf(&mut lanes);
        for (i, lane) in lanes.iter().enumerate() {
            for (j, b) in lane.to_le_bytes().iter().enumerate() {
                unsafe { *buffer.add(i * 8 + j) = *b };
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
use host::{__native_keccakf, __native_xorin};

/// The keccak state, 4-byte aligned so the precompiles can access it word-wise.
#[repr(align(4))]
#[derive(Clone)]
struct State([u8; KECCAK_WIDTH_BYTES]);

/// One rate-sized staging block, 4-byte aligned for the same reason.
#[repr(align(4))]
#[derive(Clone)]
struct Block([u8; KECCAK_RATE]);

/// Incremental keccak256 hasher.
#[derive(Clone)]
pub struct Keccak256 {
    state: State,
    /// Input staged but not yet absorbed, always `block.0[..idx]`.
    block: Block,
    idx: usize,
}

impl Keccak256 {
    pub const fn new() -> Self {
        Self {
            state: State([0u8; KECCAK_WIDTH_BYTES]),
            block: Block([0u8; KECCAK_RATE]),
            idx: 0,
        }
    }

    /// Absorb the staged block into the state and permute.
    ///
    /// `block` is always exactly [`KECCAK_RATE`] bytes here, so `len` is a multiple
    /// of 4 and both pointers are aligned by construction.
    fn absorb_block(&mut self) {
        unsafe {
            __native_xorin(
                self.state.0.as_mut_ptr(),
                self.block.0.as_ptr(),
                KECCAK_RATE,
            );
            __native_keccakf(self.state.0.as_mut_ptr());
        }
    }

    /// Absorb `input`, permuting once per full rate block.
    pub fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            let take = core::cmp::min(input.len(), KECCAK_RATE - self.idx);
            self.block.0[self.idx..self.idx + take].copy_from_slice(&input[..take]);
            self.idx += take;
            input = &input[take..];

            if self.idx == KECCAK_RATE {
                self.absorb_block();
                self.block.0.fill(0);
                self.idx = 0;
            }
        }
    }

    /// Apply pad10*1, absorb the final block, and squeeze the digest.
    pub fn finalize(mut self) -> [u8; KECCAK_OUTPUT_SIZE] {
        // Everything past `idx` is already zero, so XOR-ing the padding in is a
        // plain assignment for 0x01 and an or for 0x80 (they coincide when the
        // block is one byte short of full).
        self.block.0[self.idx] |= 0x01;
        self.block.0[KECCAK_RATE - 1] |= 0x80;
        self.absorb_block();

        let mut output = [0u8; KECCAK_OUTPUT_SIZE];
        output.copy_from_slice(&self.state.0[..KECCAK_OUTPUT_SIZE]);
        output
    }
}

impl Default for Keccak256 {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the keccak256 hash of `data`.
pub fn keccak256(data: &[u8]) -> [u8; KECCAK_OUTPUT_SIZE] {
    let mut hasher = Keccak256::new();
    hasher.update(data);
    hasher.finalize()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    extern crate std;
    use std::{vec, vec::Vec};

    use tiny_keccak::Hasher;

    use super::{KECCAK_RATE, Keccak256, keccak256};

    fn reference(data: &[u8]) -> [u8; 32] {
        let mut hasher = tiny_keccak::Keccak::v256();
        hasher.update(data);
        let mut out = [0u8; 32];
        hasher.finalize(&mut out);
        out
    }

    /// Lengths around every rate boundary, where the padding and absorb logic is
    /// most likely to be wrong: empty, one short of a block, exactly a block, one
    /// past, and the same around two and three blocks.
    #[test]
    fn matches_reference_across_block_boundaries() {
        let mut lengths: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 31, 32, 33];
        for blocks in 1..=3 {
            let n = KECCAK_RATE * blocks;
            lengths.extend([n - 2, n - 1, n, n + 1, n + 2]);
        }
        for len in lengths {
            let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            assert_eq!(
                keccak256(&data),
                reference(&data),
                "digest mismatch at len {len}"
            );
        }
    }

    /// Incremental `update` calls must produce the same digest as one shot,
    /// including when a chunk boundary lands mid-block.
    #[test]
    fn incremental_updates_match_one_shot() {
        let data: Vec<u8> = (0..(KECCAK_RATE * 2 + 45))
            .map(|i| (i % 97) as u8)
            .collect();
        for chunk in [
            1,
            3,
            4,
            7,
            64,
            KECCAK_RATE - 1,
            KECCAK_RATE,
            KECCAK_RATE + 1,
        ] {
            let mut hasher = Keccak256::new();
            for part in data.chunks(chunk) {
                hasher.update(part);
            }
            assert_eq!(
                hasher.finalize(),
                reference(&data),
                "digest mismatch with chunk size {chunk}"
            );
        }
    }
}
