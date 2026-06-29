//! Execution state with frame pointer (fp) support.
use openvm_circuit::arch::ExecutionCtxTrait;
use openvm_circuit::arch::VmExecState;
use openvm_circuit::system::memory::online::GuestMemory;
use openvm_circuit_primitives::AlignedBorrow;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use struct_reflection::StructReflection;
use struct_reflection::StructReflectionHelper;

use openvm_circuit::arch::ExecutionState as OpenVmExecutionState;

/// Like `openvm_circuit::arch::ExecutionState`, but with `fp` added.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Default, AlignedBorrow, Serialize, Deserialize, StructReflection,
)]
pub struct ExecutionState<T> {
    pub pc: T,
    pub fp: T,
    pub timestamp: T,
}

/// Maps `fp` through; OpenVM's `ExecutionState` now carries the frame pointer too.
impl<T> From<ExecutionState<T>> for OpenVmExecutionState<T> {
    fn from(state: ExecutionState<T>) -> Self {
        OpenVmExecutionState {
            pc: state.pc,
            fp: state.fp,
            timestamp: state.timestamp,
        }
    }
}

impl<T> ExecutionState<T> {
    pub fn new(pc: impl Into<T>, fp: impl Into<T>, timestamp: impl Into<T>) -> Self {
        Self {
            pc: pc.into(),
            fp: fp.into(),
            timestamp: timestamp.into(),
        }
    }

    pub fn map<U: Clone, F: Fn(T) -> U>(self, function: F) -> ExecutionState<U> {
        ExecutionState {
            pc: function(self.pc),
            fp: function(self.fp),
            timestamp: function(self.timestamp),
        }
    }
}

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
