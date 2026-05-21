#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use openvm_cuda_backend::prelude::F;
use openvm_cuda_common::{
    d_buffer::{DeviceBuffer, DeviceBufferView},
    error::CudaError,
};

/// A struct with the same memory layout as CUDA's `uint2` for FFI.
/// Vendored locally because `openvm_rv32im_circuit::cuda_abi` is `pub(crate)`
/// in openvm v2.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UInt2 {
    pub x: u32,
    pub y: u32,
}

impl UInt2 {
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
}

pub mod alu_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_alu_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range: *mut u32,
            range_bins: usize,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        unsafe {
            CudaError::from_result(_crush_alu_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
                d_range.as_mut_ptr() as *mut u32,
                range_bins,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                bitwise_num_bits,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod alu64_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_alu64_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range: *mut u32,
            range_bins: usize,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        unsafe {
            CudaError::from_result(_crush_alu64_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
                d_range.as_mut_ptr() as *mut u32,
                range_bins,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                bitwise_num_bits,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod shift_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_shift_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range: *mut u32,
            range_bins: usize,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        unsafe {
            CudaError::from_result(_crush_shift_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
                d_range.as_mut_ptr() as *mut u32,
                range_bins,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                bitwise_num_bits,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod shift64_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_shift64_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range: *mut u32,
            range_bins: usize,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        unsafe {
            CudaError::from_result(_crush_shift64_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
                d_range.as_mut_ptr() as *mut u32,
                range_bins,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                bitwise_num_bits,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod less_than_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_less_than_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        unsafe {
            CudaError::from_result(_crush_less_than_tracegen(
                d_trace.as_mut_ptr(),
                height,
                d_trace.len() / height,
                d_records.view(),
                d_range_checker.as_mut_ptr() as *mut u32,
                d_range_checker.len() as u32,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                bitwise_num_bits as u32,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod less_than64_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_less_than64_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        unsafe {
            CudaError::from_result(_crush_less_than64_tracegen(
                d_trace.as_mut_ptr(),
                height,
                d_trace.len() / height,
                d_records.view(),
                d_range_checker.as_mut_ptr() as *mut u32,
                d_range_checker.len() as u32,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                bitwise_num_bits as u32,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod mul_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_mul_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range: *mut u32,
            range_bins: usize,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: usize,
            d_range_tuple: *mut u32,
            range_tuple_sizes: UInt2,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        d_range_tuple: &DeviceBuffer<F>,
        range_tuple_sizes: UInt2,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        unsafe {
            CudaError::from_result(_crush_mul_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
                d_range.as_mut_ptr() as *mut u32,
                range_bins,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                bitwise_num_bits,
                d_range_tuple.as_mut_ptr() as *mut u32,
                range_tuple_sizes,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod mul64_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_mul64_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range: *mut u32,
            range_bins: usize,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: usize,
            d_range_tuple: *mut u32,
            range_tuple_sizes: UInt2,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        d_range_tuple: &DeviceBuffer<F>,
        range_tuple_sizes: UInt2,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        unsafe {
            CudaError::from_result(_crush_mul64_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
                d_range.as_mut_ptr() as *mut u32,
                range_bins,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                bitwise_num_bits,
                d_range_tuple.as_mut_ptr() as *mut u32,
                range_tuple_sizes,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod divrem_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_divrem_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            d_range_tuple_checker: *mut u32,
            range_tuple_checker_sizes: UInt2,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: u32,
        d_range_tuple_checker: &DeviceBuffer<F>,
        range_tuple_checker_sizes: UInt2,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        unsafe {
            CudaError::from_result(_crush_divrem_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
                d_range_checker.as_mut_ptr() as *mut u32,
                d_range_checker.len() as u32,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                bitwise_num_bits,
                d_range_tuple_checker.as_mut_ptr() as *mut u32,
                range_tuple_checker_sizes,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod eq_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_eq_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        unsafe {
            CudaError::from_result(_crush_eq_tracegen(
                d_trace.as_mut_ptr(),
                height,
                d_trace.len() / height,
                d_records.view(),
                d_range_checker.as_mut_ptr() as *mut u32,
                d_range_checker.len() as u32,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                bitwise_num_bits as u32,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod eq64_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_eq64_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        unsafe {
            CudaError::from_result(_crush_eq64_tracegen(
                d_trace.as_mut_ptr(),
                height,
                d_trace.len() / height,
                d_records.view(),
                d_range_checker.as_mut_ptr() as *mut u32,
                d_range_checker.len() as u32,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                bitwise_num_bits as u32,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod loadstore_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_load_store_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        unsafe {
            CudaError::from_result(_crush_load_store_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
                pointer_max_bits,
                d_range_checker.as_mut_ptr() as *mut u32,
                d_range_checker.len() as u32,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod load_sign_extend_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_load_sign_extend_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        unsafe {
            CudaError::from_result(_crush_load_sign_extend_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
                pointer_max_bits,
                d_range_checker.as_mut_ptr() as *mut u32,
                d_range_checker.len() as u32,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod call_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_call_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            pointer_max_bits: usize,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        width: usize,
        d_records: &DeviceBuffer<u8>,
        pointer_max_bits: usize,
        d_range_checker: &DeviceBuffer<F>,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        unsafe {
            CudaError::from_result(_crush_call_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
                pointer_max_bits,
                d_range_checker.as_mut_ptr() as *mut u32,
                d_range_checker.len() as u32,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod jump_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_jump_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_bins: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        range_checker_bins: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        unsafe {
            CudaError::from_result(_crush_jump_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
                d_range_checker.as_mut_ptr() as *mut u32,
                range_checker_bins,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod const32_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_const32_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range: *mut u32,
            range_bins: usize,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: usize,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range: &DeviceBuffer<F>,
        range_bins: usize,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: usize,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        unsafe {
            CudaError::from_result(_crush_const32_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
                d_range.as_mut_ptr() as *mut u32,
                range_bins,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                bitwise_num_bits,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod hintstore_cuda {
    use super::{super::hintstore::OffsetInfo, *};

    unsafe extern "C" {
        pub fn _crush_hintstore_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: *const u8,
            rows_used: usize,
            d_record_offsets: *const OffsetInfo,
            pointer_max_bits: u32,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        rows_used: usize,
        d_record_offsets: &DeviceBuffer<OffsetInfo>,
        pointer_max_bits: u32,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: u32,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        unsafe {
            CudaError::from_result(_crush_hintstore_tracegen(
                d_trace.as_mut_ptr(),
                height,
                d_trace.len() / height,
                d_records.as_ptr(),
                rows_used,
                d_record_offsets.as_ptr(),
                pointer_max_bits,
                d_range_checker.as_mut_ptr() as *mut u32,
                d_range_checker.len() as u32,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                bitwise_num_bits,
                timestamp_max_bits,
            ))
        }
    }
}

pub mod divrem64_cuda {
    use super::*;
    unsafe extern "C" {
        fn _crush_divrem64_tracegen(
            d_trace: *mut F,
            height: usize,
            width: usize,
            d_records: DeviceBufferView,
            d_range_checker: *mut u32,
            range_checker_num_bins: u32,
            d_bitwise_lookup: *mut u32,
            bitwise_num_bits: u32,
            d_range_tuple_checker: *mut u32,
            range_tuple_checker_sizes: UInt2,
            timestamp_max_bits: u32,
        ) -> i32;
    }

    pub unsafe fn tracegen(
        d_trace: &DeviceBuffer<F>,
        height: usize,
        d_records: &DeviceBuffer<u8>,
        d_range_checker: &DeviceBuffer<F>,
        d_bitwise_lookup: &DeviceBuffer<F>,
        bitwise_num_bits: u32,
        d_range_tuple_checker: &DeviceBuffer<F>,
        range_tuple_checker_sizes: UInt2,
        timestamp_max_bits: u32,
    ) -> Result<(), CudaError> {
        let width = d_trace.len() / height;
        unsafe {
            CudaError::from_result(_crush_divrem64_tracegen(
                d_trace.as_mut_ptr(),
                height,
                width,
                d_records.view(),
                d_range_checker.as_mut_ptr() as *mut u32,
                d_range_checker.len() as u32,
                d_bitwise_lookup.as_mut_ptr() as *mut u32,
                bitwise_num_bits,
                d_range_tuple_checker.as_mut_ptr() as *mut u32,
                range_tuple_checker_sizes,
                timestamp_max_bits,
            ))
        }
    }
}
