#![no_std]

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// Compute keccak256 hash: reads `len` bytes from `input`, writes 32-byte hash to `output`.
    /// Pointers must be 4-byte aligned.
    pub unsafe fn __native_keccak256(input: *const u8, len: usize, output: *mut u8);
}

/// Compute keccak256 hash of `data` using native precompile.
pub fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut output = [0u8; 32];
    unsafe {
        __native_keccak256(data.as_ptr(), data.len(), output.as_mut_ptr());
    }
    output
}
