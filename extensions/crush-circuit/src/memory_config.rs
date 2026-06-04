use openvm_circuit::{
    arch::{ADDR_SPACE_OFFSET, DEFAULT_MAX_NUM_PUBLIC_VALUES, MemoryConfig},
    system::memory::{
        POINTER_MAX_BITS, merkle::public_values::PUBLIC_VALUES_AS, online::GuestMemory,
    },
};
use openvm_instructions::riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS};
use openvm_stark_backend::p3_field::PrimeField32;

/// This address space is only used for execution, to store the frame pointer at address 0.
pub const FP_AS: u32 = 5;

/// Same as `MemoryConfig::default()`, but:
/// - Register address space has the same size as the memory address space.
/// - Removed native address space (only needed in recursion).
/// - Added address space for FP. It has one cell to store the current FP value.
pub fn memory_config_with_fp() -> MemoryConfig {
    let mut addr_spaces =
        MemoryConfig::empty_address_space_configs((1 << 3) + ADDR_SPACE_OFFSET as usize);
    const MAX_CELLS: usize = 1 << 29;
    addr_spaces[RV32_REGISTER_AS as usize].num_cells = MAX_CELLS;
    addr_spaces[RV32_MEMORY_AS as usize].num_cells = MAX_CELLS;
    addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = DEFAULT_MAX_NUM_PUBLIC_VALUES;
    // GPU requires num_cells to be divisible by DIGEST_WIDTH (8).
    // We only need 4 bytes (size_of::<u32>()), but round up to 8.
    addr_spaces[FP_AS as usize].num_cells = 8;
    MemoryConfig::new(3, addr_spaces, POINTER_MAX_BITS, 29, 17)
}

/// Utility trait to read the frame pointer from memory, used in preflight and execution.
/// Methods are generic over `F` because FP_AS uses `native32` cell type, which stores
/// field elements (not raw u32). The type must match the ZK backend's base field.
pub trait FpMemory {
    fn fp<F: PrimeField32>(&self) -> u32;

    #[allow(dead_code)]
    fn set_fp<F: PrimeField32>(&mut self, value: u32);
}

impl FpMemory for GuestMemory {
    fn fp<F: PrimeField32>(&self) -> u32 {
        // SAFETY: FP_AS uses native32 cell type (F), so T=F is the correct type.
        // Block size 4 matches v2's DEFAULT_BLOCK_SIZE; only cell 0 carries the FP.
        unsafe { self.read::<F, 4>(FP_AS, 0)[0].as_canonical_u32() }
    }

    fn set_fp<F: PrimeField32>(&mut self, value: u32) {
        // SAFETY: FP_AS uses native32 cell type (F), so T=F is the correct type.
        // Pads cells 1..4 with zero so the 4-cell block stays canonical.
        unsafe { self.write::<F, 4>(FP_AS, 0, [F::from_u32(value), F::ZERO, F::ZERO, F::ZERO]) }
    }
}
