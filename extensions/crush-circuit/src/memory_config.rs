use openvm_circuit::arch::{ADDR_SPACE_OFFSET, DEFAULT_MAX_NUM_PUBLIC_VALUES, MemoryConfig};
use openvm_circuit::system::memory::{POINTER_MAX_BITS, merkle::public_values::PUBLIC_VALUES_AS};
use openvm_instructions::riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS};

/// Same as `MemoryConfig::default()`, but:
/// - Register address space has the same size as the memory address space.
/// - Removed native address space (only needed in recursion).
///
/// The frame pointer is no longer stored in a memory cell; it is part of the VM
/// `ExecutionState` (carried on the execution bus like `pc`), so no FP address space is needed.
pub fn crush_memory_config() -> MemoryConfig {
    let mut addr_spaces =
        MemoryConfig::empty_address_space_configs((1 << 3) + ADDR_SPACE_OFFSET as usize);
    const MAX_CELLS: usize = 1 << 29;
    addr_spaces[RV32_REGISTER_AS as usize].num_cells = MAX_CELLS;
    addr_spaces[RV32_MEMORY_AS as usize].num_cells = MAX_CELLS;
    addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = DEFAULT_MAX_NUM_PUBLIC_VALUES;
    MemoryConfig::new(3, addr_spaces, POINTER_MAX_BITS, 29, 17)
}
