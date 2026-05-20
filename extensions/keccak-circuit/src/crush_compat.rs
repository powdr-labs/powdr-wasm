//! Types and functions from crush-circuit that keccak-circuit needs,
//! inlined here to avoid a circular dependency.

pub mod adapters {
    use openvm_circuit::system::memory::{MemoryAddress, online::TracingMemory};
    use openvm_instructions::riscv::RV32_REGISTER_AS;
    use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};

    use super::memory_config::FP_AS;

    #[inline(always)]
    pub fn fp_addr<F: FieldAlgebra>() -> MemoryAddress<F, F> {
        MemoryAddress::new(F::from_canonical_u32(FP_AS), F::ZERO)
    }

    #[inline(always)]
    pub fn reg_addr<F: FieldAlgebra>(ptr: F) -> MemoryAddress<F, F> {
        MemoryAddress::new(F::from_canonical_u32(RV32_REGISTER_AS), ptr)
    }

    /// Tracing read of the frame pointer from FP_AS address 0.
    /// Returns the fp value and records the previous timestamp for trace generation.
    #[inline(always)]
    pub fn tracing_read_fp<F: PrimeField32>(
        memory: &mut TracingMemory,
        prev_timestamp: &mut u32,
    ) -> u32 {
        // SAFETY: FP_AS uses native32 cell type (F), block size 1, align 1.
        let (t_prev, data) = unsafe { memory.read::<F, 1, 1>(FP_AS, 0) };
        *prev_timestamp = t_prev;
        data[0].as_canonical_u32()
    }
}

pub mod execution {
    use openvm_circuit::arch::ExecutionState as OpenVmExecutionState;
    use openvm_circuit_primitives::AlignedBorrow;
    use serde::{Deserialize, Serialize};
    use struct_reflection::StructReflection;
    use struct_reflection::StructReflectionHelper;

    /// Like `openvm_circuit::arch::ExecutionState`, but with `fp` added.
    #[repr(C)]
    #[derive(
        Clone,
        Copy,
        Debug,
        PartialEq,
        Default,
        AlignedBorrow,
        Serialize,
        Deserialize,
        StructReflection,
    )]
    pub struct ExecutionState<T> {
        pub pc: T,
        pub fp: T,
        pub timestamp: T,
    }

    /// Discards `fp` when converting to `OpenVmExecutionState`.
    impl<T> From<ExecutionState<T>> for OpenVmExecutionState<T> {
        fn from(state: ExecutionState<T>) -> Self {
            OpenVmExecutionState {
                pc: state.pc,
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
    }
}

pub mod memory_config {
    use openvm_circuit::system::memory::online::GuestMemory;
    use openvm_stark_backend::p3_field::PrimeField32;

    /// This address space is only used for execution, to store the frame pointer at address 0.
    pub const FP_AS: u32 = 5;

    /// Utility trait to read the frame pointer from memory, used in preflight and execution.
    /// Methods are generic over `F` because FP_AS uses `native32` cell type, which stores
    /// field elements (not raw u32). The type must match the ZK backend's base field.
    pub trait FpMemory {
        fn fp<F: PrimeField32>(&self) -> u32;
    }

    impl FpMemory for GuestMemory {
        fn fp<F: PrimeField32>(&self) -> u32 {
            // SAFETY: FP_AS uses native32 cell type (F), so T=F is the correct type.
            unsafe { self.read::<F, 1>(FP_AS, 0)[0].as_canonical_u32() }
        }
    }
}
