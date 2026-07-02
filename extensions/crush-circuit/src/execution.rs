//! Frame-pointer-aware register read/write helpers for crush execution.
//!
//! The frame pointer is no longer a separate wrapper type: it is carried in openvm's
//! `ExecutionState` / `VmState` as `extra_regs[0]` (see `openvm_circuit::arch::EXTRA_EXEC_REGS`).
use openvm_circuit::arch::ExecutionCtxTrait;
use openvm_circuit::arch::VmExecState;
use openvm_circuit::system::memory::online::GuestMemory;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_stark_backend::p3_field::PrimeField32;

/// Reads `NUM_LIMBS` bytes by doing `NUM_REG_OPS` reads of `RV32_REGISTER_NUM_LIMBS` bytes.
/// This matches the behavior enforced by the adapter constraints, which read register-sized
/// chunks rather than a single large block.
pub fn vm_read_multiple_ops<
    const NUM_LIMBS: usize,
    const NUM_REG_OPS: usize,
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
>(
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
    addr_space: u32,
    ptr: u32,
) -> [u8; NUM_LIMBS] {
    const { assert!(NUM_LIMBS == NUM_REG_OPS * RV32_REGISTER_NUM_LIMBS) };
    let mut buf = [0u8; NUM_LIMBS];
    for i in 0..NUM_REG_OPS {
        let offset = i * RV32_REGISTER_NUM_LIMBS;
        let chunk: [u8; RV32_REGISTER_NUM_LIMBS] =
            exec_state.vm_read(addr_space, ptr + offset as u32);
        buf[offset..offset + RV32_REGISTER_NUM_LIMBS].copy_from_slice(&chunk);
    }
    buf
}

/// Writes `NUM_LIMBS` bytes by doing `NUM_REG_OPS` writes of `RV32_REGISTER_NUM_LIMBS` bytes.
/// This matches the behavior enforced by the adapter constraints.
pub fn vm_write_multiple_ops<
    const NUM_LIMBS: usize,
    const NUM_REG_OPS: usize,
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
>(
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
    addr_space: u32,
    ptr: u32,
    data: &[u8; NUM_LIMBS],
) {
    const { assert!(NUM_LIMBS == NUM_REG_OPS * RV32_REGISTER_NUM_LIMBS) };
    for i in 0..NUM_REG_OPS {
        let offset = i * RV32_REGISTER_NUM_LIMBS;
        let chunk: [u8; RV32_REGISTER_NUM_LIMBS] = std::array::from_fn(|j| data[offset + j]);
        exec_state.vm_write(addr_space, ptr + offset as u32, &chunk);
    }
}
