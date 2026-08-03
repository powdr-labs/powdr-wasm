//! SHA-256 and SHA-512 for crush guests, built on the `SHA256`/`SHA512` compression
//! precompiles.
//!
//! Each precompile performs exactly one compression: given the previous hash state and
//! one message block, it writes the new state. Everything else — the initial state, the
//! block loop, and the padding — is ordinary guest code here, mirroring how OpenVM's own
//! sha2 guest drives the same opcodes.
//!
//! Input is staged in a block-sized, 4-byte-aligned buffer, so every precompile call gets
//! a full block from an aligned pointer regardless of how the caller's input is aligned or
//! sized.
//!
//! The two variants differ only in word size, block size, initial state, and the width of
//! the trailing length field, so both hashers are generated from one implementation by
//! `sha2_hasher!`. The block loop and the padding — the parts most likely to be wrong, and
//! where a bug was in fact caught during development — therefore exist once.

#![no_std]

/// Bytes per SHA-256 message block.
pub const SHA256_BLOCK_BYTES: usize = 64;
/// Bytes in the SHA-256 state (8 x u32).
pub const SHA256_STATE_BYTES: usize = 32;
/// Bytes in a SHA-256 digest.
pub const SHA256_OUTPUT_SIZE: usize = 32;

/// Bytes per SHA-512 message block.
pub const SHA512_BLOCK_BYTES: usize = 128;
/// Bytes in the SHA-512 state (8 x u64).
pub const SHA512_STATE_BYTES: usize = 64;
/// Bytes in a SHA-512 digest.
pub const SHA512_OUTPUT_SIZE: usize = 64;

/// SHA-256 initial hash value (FIPS 180-4 section 5.3.3).
const H0_256: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// SHA-512 initial hash value (FIPS 180-4 section 5.3.5).
const H0_512: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// One SHA-256 compression. `state` and `output` are 32 bytes (8 little-endian u32
    /// words), `input` is one 64-byte message block. All three must be 4-byte aligned.
    /// `output` may alias `state`.
    unsafe fn __sha256_compress(state: *const u8, input: *const u8, output: *mut u8);

    /// One SHA-512 compression. `state` and `output` are 64 bytes (8 little-endian u64
    /// words), `input` is one 128-byte message block. All three must be 4-byte aligned.
    /// `output` may alias `state`.
    unsafe fn __sha512_compress(state: *const u8, input: *const u8, output: *mut u8);
}

/// Software stand-ins for the precompiles, so the block loop and padding below can be
/// exercised off target. That logic is plain guest code rather than constrained circuitry,
/// so it is worth testing directly; see the tests at the bottom.
#[cfg(not(target_arch = "wasm32"))]
mod host {
    /// Generate a software compression matching the precompile's state encoding: 8
    /// little-endian `$word` values, which is already the H value the reference
    /// compression function consumes.
    macro_rules! host_compress {
        ($name:ident, $word:ty, $block:expr, $compress:path) => {
            pub(super) unsafe fn $name(state: *const u8, input: *const u8, output: *mut u8) {
                const W: usize = size_of::<$word>();
                let mut words = [0 as $word; 8];
                for (i, w) in words.iter_mut().enumerate() {
                    let mut b = [0u8; W];
                    for (j, x) in b.iter_mut().enumerate() {
                        *x = unsafe { *state.add(i * W + j) };
                    }
                    *w = <$word>::from_le_bytes(b);
                }
                let mut block = [0u8; $block];
                for (i, b) in block.iter_mut().enumerate() {
                    *b = unsafe { *input.add(i) };
                }
                $compress(&mut words, &[block.into()]);
                for (i, w) in words.iter().enumerate() {
                    for (j, b) in w.to_le_bytes().iter().enumerate() {
                        unsafe { *output.add(i * W + j) = *b };
                    }
                }
            }
        };
    }

    host_compress!(__sha256_compress, u32, 64, sha2::compress256);
    host_compress!(__sha512_compress, u64, 128, sha2::compress512);
}

#[cfg(not(target_arch = "wasm32"))]
use host::{__sha256_compress, __sha512_compress};

/// A byte buffer aligned to 4 bytes, which the precompiles require of every pointer.
#[repr(align(4))]
#[derive(Clone)]
struct Aligned4<const N: usize>([u8; N]);

/// Generate a SHA-2 hasher over one of the compression precompiles.
///
/// `$len_bytes` is the width of the trailing big-endian bit-length field: 8 for SHA-256,
/// 16 for SHA-512. The message length is tracked as a `u64` either way, which is ample --
/// SHA-512 permits a 128-bit length, but no guest will hash 2^61 bytes.
macro_rules! sha2_hasher {
    (
        $(#[$meta:meta])*
        $name:ident, $word:ty, $h0:expr, $block:expr, $state_bytes:expr,
        $out:expr, $len_bytes:expr, $compress:ident, $fn_name:ident, $fn_doc:expr
    ) => {
        $(#[$meta])*
        #[derive(Clone)]
        pub struct $name {
            /// The hash state as 8 little-endian `$word` values.
            state: Aligned4<$state_bytes>,
            /// Input staged but not yet compressed, always `block.0[..idx]`.
            block: Aligned4<$block>,
            idx: usize,
            /// Total message length in bytes, for the length field in the padding.
            len: u64,
        }

        impl $name {
            pub fn new() -> Self {
                let mut state = Aligned4([0u8; $state_bytes]);
                for (chunk, w) in state.0.chunks_exact_mut(size_of::<$word>()).zip($h0) {
                    chunk.copy_from_slice(&w.to_le_bytes());
                }
                Self {
                    state,
                    block: Aligned4([0u8; $block]),
                    idx: 0,
                    len: 0,
                }
            }

            /// Compress the staged block into the state.
            ///
            /// `block` is always exactly `$block` bytes here and both buffers are aligned
            /// by construction. `output` aliases `state`, which the precompile allows.
            fn compress_block(&mut self) {
                unsafe {
                    $compress(
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
                    let take = core::cmp::min(input.len(), $block - self.idx);
                    self.block.0[self.idx..self.idx + take].copy_from_slice(&input[..take]);
                    self.idx += take;
                    input = &input[take..];

                    if self.idx == $block {
                        self.compress_block();
                        self.idx = 0;
                    }
                }
            }

            /// Apply SHA-2 padding (0x80, zeroes, then the big-endian bit length),
            /// compress the final block(s), and return the digest.
            pub fn finalize(mut self) -> [u8; $out] {
                let bit_len = self.len.wrapping_mul(8);

                // 0x80 terminator, then zeroes.
                self.block.0[self.idx] = 0x80;
                self.idx += 1;
                self.block.0[self.idx..].fill(0);

                // The length field must fit in this block; otherwise emit a full extra one.
                if self.idx > $block - $len_bytes {
                    self.compress_block();
                    self.block.0.fill(0);
                }
                // Only the low 64 bits can be non-zero, so the upper half of a 16-byte
                // field stays zero and writing the last 8 bytes is enough.
                self.block.0[$block - 8..].copy_from_slice(&bit_len.to_be_bytes());
                self.compress_block();

                // The state is little-endian words; the digest is those words big-endian.
                let mut out = [0u8; $out];
                for (o, s) in out
                    .chunks_exact_mut(size_of::<$word>())
                    .zip(self.state.0.chunks_exact(size_of::<$word>()))
                {
                    let w = <$word>::from_le_bytes(s.try_into().unwrap());
                    o.copy_from_slice(&w.to_be_bytes());
                }
                out
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        #[doc = $fn_doc]
        pub fn $fn_name(data: &[u8]) -> [u8; $out] {
            let mut hasher = $name::new();
            hasher.update(data);
            hasher.finalize()
        }
    };
}

sha2_hasher!(
    /// Incremental SHA-256 hasher.
    Sha256,
    u32,
    H0_256,
    SHA256_BLOCK_BYTES,
    SHA256_STATE_BYTES,
    SHA256_OUTPUT_SIZE,
    8,
    __sha256_compress,
    sha256,
    "Compute the SHA-256 hash of `data`."
);

sha2_hasher!(
    /// Incremental SHA-512 hasher.
    Sha512,
    u64,
    H0_512,
    SHA512_BLOCK_BYTES,
    SHA512_STATE_BYTES,
    SHA512_OUTPUT_SIZE,
    16,
    __sha512_compress,
    sha512,
    "Compute the SHA-512 hash of `data`."
);

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    extern crate std;
    use std::{string::String, vec, vec::Vec};

    use sha2::Digest;

    use super::{SHA256_BLOCK_BYTES, SHA512_BLOCK_BYTES, Sha256, Sha512, sha256, sha512};

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| std::format!("{b:02x}")).collect()
    }

    fn reference_256(data: &[u8]) -> [u8; 32] {
        let mut h = sha2::Sha256::new();
        h.update(data);
        h.finalize().into()
    }

    fn reference_512(data: &[u8]) -> [u8; 64] {
        let mut h = sha2::Sha512::new();
        h.update(data);
        h.finalize().into()
    }

    /// The canonical FIPS vectors, as a check that the state encoding and the digest byte
    /// order are right and not merely self-consistent.
    #[test]
    fn matches_known_abc_vectors() {
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(&sha512(b"abc")),
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a\
             2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
        );
    }

    /// Lengths around every block boundary, where the padding is most likely to be wrong.
    /// The interesting window is the one where the length field no longer fits and an
    /// extra block is required: 56..64 for SHA-256, 112..128 for SHA-512.
    fn boundary_lengths(block: usize, len_field: usize) -> Vec<usize> {
        let mut v: Vec<usize> = vec![0, 1, 2, 3, 4];
        for blocks in 0..=2 {
            let n = block * blocks;
            let tight = block - len_field;
            v.extend([
                n + tight - 2,
                n + tight - 1,
                n + tight,
                n + tight + 1,
                n + block - 2,
                n + block - 1,
                n + block,
                n + block + 1,
            ]);
        }
        v
    }

    #[test]
    fn sha256_matches_reference_across_block_boundaries() {
        for len in boundary_lengths(SHA256_BLOCK_BYTES, 8) {
            let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            assert_eq!(
                sha256(&data),
                reference_256(&data),
                "sha256 mismatch at len {len}"
            );
        }
    }

    #[test]
    fn sha512_matches_reference_across_block_boundaries() {
        for len in boundary_lengths(SHA512_BLOCK_BYTES, 16) {
            let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            assert_eq!(
                sha512(&data),
                reference_512(&data),
                "sha512 mismatch at len {len}"
            );
        }
    }

    /// Incremental `update` calls must produce the same digest as one shot, including when
    /// a chunk boundary lands mid-block for either variant.
    #[test]
    fn incremental_updates_match_one_shot() {
        let data: Vec<u8> = (0..(SHA512_BLOCK_BYTES * 3 + 17))
            .map(|i| (i % 97) as u8)
            .collect();

        for chunk in [1, 3, 4, 17, 63, 64, 65, 127, 128, 129] {
            let mut h256 = Sha256::new();
            let mut h512 = Sha512::new();
            for part in data.chunks(chunk) {
                h256.update(part);
                h512.update(part);
            }
            assert_eq!(
                h256.finalize(),
                reference_256(&data),
                "sha256 mismatch with chunk size {chunk}"
            );
            assert_eq!(
                h512.finalize(),
                reference_512(&data),
                "sha512 mismatch with chunk size {chunk}"
            );
        }
    }
}
