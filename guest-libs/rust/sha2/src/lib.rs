//! SHA-256 for crush guests, built on the `SHA256` compression precompile.
//!
//! The precompile performs exactly one compression: given the previous hash state and
//! one 64-byte message block, it writes the new state. Everything else — the initial
//! state, the block loop, and the padding — is ordinary guest code here, mirroring how
//! OpenVM's own sha2 guest drives the same opcode.
//!
//! Input is staged in a block-sized, 4-byte-aligned buffer, so every precompile call
//! gets a full 64-byte block from an aligned pointer regardless of how the caller's
//! input is aligned or sized.

#![no_std]

/// Bytes per SHA-256 message block.
pub const SHA256_BLOCK_BYTES: usize = 64;
/// Bytes in the SHA-256 state (8 x u32).
pub const SHA256_STATE_BYTES: usize = 32;
/// Bytes in a SHA-256 digest.
pub const SHA256_OUTPUT_SIZE: usize = 32;

/// SHA-256 initial hash value (FIPS 180-4 section 5.3.3).
const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// One SHA-256 compression. `state` and `output` are 32 bytes (8 little-endian
    /// u32 words), `input` is one 64-byte message block. All three must be 4-byte
    /// aligned. `output` may alias `state`.
    unsafe fn __native_sha256_compress(state: *const u8, input: *const u8, output: *mut u8);
}

/// Software stand-in for the precompile, so the padding and block logic below can be
/// exercised off target. That logic is plain guest code rather than constrained
/// circuitry, so it is worth testing directly; see the tests at the bottom.
#[cfg(not(target_arch = "wasm32"))]
unsafe fn __native_sha256_compress(state: *const u8, input: *const u8, output: *mut u8) {
    let mut words = [0u32; 8];
    for (i, w) in words.iter_mut().enumerate() {
        let mut b = [0u8; 4];
        for (j, x) in b.iter_mut().enumerate() {
            *x = unsafe { *state.add(i * 4 + j) };
        }
        // The precompile stores the state as little-endian u32 words, which is exactly
        // the H value that compress256 operates on.
        *w = u32::from_le_bytes(b);
    }
    let mut block = [0u8; SHA256_BLOCK_BYTES];
    for (i, b) in block.iter_mut().enumerate() {
        *b = unsafe { *input.add(i) };
    }
    sha2::compress256(&mut words, &[block.into()]);
    for (i, w) in words.iter().enumerate() {
        for (j, b) in w.to_le_bytes().iter().enumerate() {
            unsafe { *output.add(i * 4 + j) = *b };
        }
    }
}

/// The hash state, 4-byte aligned so the precompile can access it word-wise.
#[repr(align(4))]
#[derive(Clone)]
struct State([u8; SHA256_STATE_BYTES]);

/// One block-sized staging buffer, 4-byte aligned for the same reason.
#[repr(align(4))]
#[derive(Clone)]
struct Block([u8; SHA256_BLOCK_BYTES]);

/// Incremental SHA-256 hasher.
#[derive(Clone)]
pub struct Sha256 {
    state: State,
    /// Input staged but not yet compressed, always `block.0[..idx]`.
    block: Block,
    idx: usize,
    /// Total message length in bytes, for the length suffix in the padding.
    len: u64,
}

impl Sha256 {
    pub fn new() -> Self {
        let mut state = State([0u8; SHA256_STATE_BYTES]);
        for (chunk, w) in state.0.chunks_exact_mut(4).zip(H0) {
            chunk.copy_from_slice(&w.to_le_bytes());
        }
        Self {
            state,
            block: Block([0u8; SHA256_BLOCK_BYTES]),
            idx: 0,
            len: 0,
        }
    }

    /// Compress the staged block into the state.
    ///
    /// `block` is always exactly [`SHA256_BLOCK_BYTES`] here and both buffers are
    /// aligned by construction. `output` aliases `state`, which the precompile allows.
    fn compress_block(&mut self) {
        unsafe {
            __native_sha256_compress(
                self.state.0.as_ptr(),
                self.block.0.as_ptr(),
                self.state.0.as_mut_ptr(),
            );
        }
    }

    /// Absorb `input`, compressing once per full block.
    pub fn update(&mut self, mut input: &[u8]) {
        self.len = self.len.wrapping_add(input.len() as u64);
        while !input.is_empty() {
            let take = core::cmp::min(input.len(), SHA256_BLOCK_BYTES - self.idx);
            self.block.0[self.idx..self.idx + take].copy_from_slice(&input[..take]);
            self.idx += take;
            input = &input[take..];

            if self.idx == SHA256_BLOCK_BYTES {
                self.compress_block();
                self.idx = 0;
            }
        }
    }

    /// Apply SHA-2 padding (0x80, zeroes, then the 64-bit big-endian bit length),
    /// compress the final block(s), and return the digest.
    pub fn finalize(mut self) -> [u8; SHA256_OUTPUT_SIZE] {
        let bit_len = self.len.wrapping_mul(8);

        // 0x80 terminator, then zeroes.
        self.block.0[self.idx] = 0x80;
        self.idx += 1;
        self.block.0[self.idx..].fill(0);

        // The 8-byte length must fit in this block; otherwise emit a full extra block.
        if self.idx > SHA256_BLOCK_BYTES - 8 {
            self.compress_block();
            self.block.0.fill(0);
        }
        self.block.0[SHA256_BLOCK_BYTES - 8..].copy_from_slice(&bit_len.to_be_bytes());
        self.compress_block();

        // The state is little-endian u32 words; the digest is those words big-endian.
        let mut out = [0u8; SHA256_OUTPUT_SIZE];
        for (o, s) in out.chunks_exact_mut(4).zip(self.state.0.chunks_exact(4)) {
            let w = u32::from_le_bytes(s.try_into().unwrap());
            o.copy_from_slice(&w.to_be_bytes());
        }
        out
    }
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the SHA-256 hash of `data`.
pub fn sha256(data: &[u8]) -> [u8; SHA256_OUTPUT_SIZE] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    extern crate std;
    use std::{vec, vec::Vec};

    use sha2::Digest;

    use super::{SHA256_BLOCK_BYTES, Sha256, sha256};

    fn reference(data: &[u8]) -> [u8; 32] {
        let mut h = sha2::Sha256::new();
        h.update(data);
        h.finalize().into()
    }

    /// The canonical FIPS test vector, as a check that the state encoding and the
    /// digest byte order are right and not merely self-consistent.
    #[test]
    fn matches_known_abc_vector() {
        let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let got = sha256(b"abc");
        let hex: std::string::String = got.iter().map(|b| std::format!("{b:02x}")).collect();
        assert_eq!(hex, expected);
    }

    /// Lengths around every block boundary, where the padding is most likely to be
    /// wrong -- in particular 56..64, where the length suffix no longer fits and an
    /// extra block is required.
    #[test]
    fn matches_reference_across_block_boundaries() {
        let mut lengths: Vec<usize> = vec![0, 1, 2, 3, 4, 31, 32, 33];
        for blocks in 0..=2 {
            let n = SHA256_BLOCK_BYTES * blocks;
            lengths.extend([
                n + 54,
                n + 55,
                n + 56,
                n + 57,
                n + 62,
                n + 63,
                n + SHA256_BLOCK_BYTES,
                n + SHA256_BLOCK_BYTES + 1,
            ]);
        }
        for len in lengths {
            let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            assert_eq!(
                sha256(&data),
                reference(&data),
                "digest mismatch at len {len}"
            );
        }
    }

    /// Incremental `update` calls must produce the same digest as one shot,
    /// including when a chunk boundary lands mid-block.
    #[test]
    fn incremental_updates_match_one_shot() {
        let data: Vec<u8> = (0..(SHA256_BLOCK_BYTES * 3 + 17))
            .map(|i| (i % 97) as u8)
            .collect();
        for chunk in [
            1,
            3,
            4,
            17,
            SHA256_BLOCK_BYTES - 1,
            SHA256_BLOCK_BYTES,
            SHA256_BLOCK_BYTES + 1,
        ] {
            let mut hasher = Sha256::new();
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
