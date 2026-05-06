// Call adapter CUDA implementation for GPU tracegen.
// Handles the 6 memory operations (reads/writes) and carry-chain arithmetic
// for the Call/CallIndirect/Ret instructions.
#pragma once

#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"
#include "crush/execution.cuh"

using namespace riscv;

// Mirror of CallAdapterCols<T> in Rust (adapters/call.rs)
template <typename T> struct CrushCallAdapterCols {
    CrushExecutionState<T> from_state;

    T to_fp_operand;
    T save_fp_ptr;
    T save_pc_ptr;
    T to_pc_operand;

    MemoryReadAuxCols<T> fp_read_aux;
    MemoryReadAuxCols<T> to_fp_read_aux;
    MemoryReadAuxCols<T> to_pc_read_aux;
    MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS> save_fp_write_aux;
    MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS> save_pc_write_aux;
    // FP_AS write: prev_data is 4 field elements (DEFAULT_BLOCK_SIZE for v2 —
    // each cell stores a `field32` value; only cell 0 carries the FP).
    MemoryWriteAuxCols<T, 4> fp_write_aux;

    T offset_limbs[2];
    T new_fp_limbs[2];
};

// Mirror of CallAdapterRecord in Rust
struct CrushCallAdapterRecord {
    uint32_t from_pc;
    uint32_t fp;
    uint32_t from_timestamp;

    uint32_t to_fp_operand;
    uint32_t save_fp_ptr;
    uint32_t save_pc_ptr;
    uint32_t to_pc_operand;

    bool has_pc_read;
    bool has_save;

    MemoryReadAuxRecord fp_read_aux;
    MemoryReadAuxRecord to_fp_read_aux;
    MemoryReadAuxRecord to_pc_read_aux;
    MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS> save_fp_write_aux;
    MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS> save_pc_write_aux;
    // FP_AS write: prev_data stored as canonical u32 (not Montgomery-encoded Fp).
    // 4 cells (DEFAULT_BLOCK_SIZE) — only cell 0 carries the FP, cells 1..4 are zero.
    MemoryWriteAuxRecord<uint32_t, 4> fp_write_aux;
};

struct CrushCallAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    template <typename T>
    using Cols = CrushCallAdapterCols<T>;

    __device__ CrushCallAdapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, CrushCallAdapterRecord record) {
        bool has_save = record.has_save;
        bool has_pc_read = record.has_pc_read;
        bool has_fp_read = !has_save;
        uint32_t fp = record.fp;
        uint32_t to_fp_operand = record.to_fp_operand;

        // Scalar fields
        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Cols, from_state.fp, fp);
        COL_WRITE_VALUE(row, Cols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Cols, to_fp_operand, to_fp_operand);
        COL_WRITE_VALUE(row, Cols, save_fp_ptr, record.save_fp_ptr);
        COL_WRITE_VALUE(row, Cols, save_pc_ptr, record.save_pc_ptr);
        COL_WRITE_VALUE(row, Cols, to_pc_operand, record.to_pc_operand);

        // Fill in reverse timestamp order to match Rust filler

        // 5. FP write (native32 cell type: prev_data is 4 field elements;
        // cell 0 carries the FP, cells 1..4 are zero).
        // Record stores each cell as canonical u32; must convert to Montgomery Fp.
        {
            uint32_t timestamp = record.from_timestamp + 5;
            size_t base = COL_INDEX(Cols, fp_write_aux);
            using FpWriteAux = MemoryWriteAuxCols<uint8_t, 4>;
            for (size_t i = 0; i < 4; i++) {
                row.write(
                    base + offsetof(FpWriteAux, prev_data) + i,
                    Fp(record.fp_write_aux.prev_data[i])
                );
            }
            mem_helper.fill(
                row.slice_from(base),
                record.fp_write_aux.prev_timestamp,
                timestamp
            );
        }

        // 4. save_pc write (conditional on has_save)
        {
            uint32_t timestamp = record.from_timestamp + 4;
            using WriteAux = MemoryWriteAuxCols<uint8_t, RV32_REGISTER_NUM_LIMBS>;
            size_t base = COL_INDEX(Cols, save_pc_write_aux);
            if (has_save) {
                row.write_array(
                    base + offsetof(WriteAux, prev_data),
                    RV32_REGISTER_NUM_LIMBS,
                    record.save_pc_write_aux.prev_data
                );
                mem_helper.fill(
                    row.slice_from(base),
                    record.save_pc_write_aux.prev_timestamp,
                    timestamp
                );
            } else {
                mem_helper.fill_zero(row.slice_from(base));
                // Also zero out prev_data
                for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
                    row.write(base + offsetof(WriteAux, prev_data) + i, 0);
                }
            }
        }

        // 3. save_fp write (conditional on has_save)
        {
            uint32_t timestamp = record.from_timestamp + 3;
            using WriteAux = MemoryWriteAuxCols<uint8_t, RV32_REGISTER_NUM_LIMBS>;
            size_t base = COL_INDEX(Cols, save_fp_write_aux);
            if (has_save) {
                row.write_array(
                    base + offsetof(WriteAux, prev_data),
                    RV32_REGISTER_NUM_LIMBS,
                    record.save_fp_write_aux.prev_data
                );
                mem_helper.fill(
                    row.slice_from(base),
                    record.save_fp_write_aux.prev_timestamp,
                    timestamp
                );
            } else {
                mem_helper.fill_zero(row.slice_from(base));
                for (size_t i = 0; i < RV32_REGISTER_NUM_LIMBS; i++) {
                    row.write(base + offsetof(WriteAux, prev_data) + i, 0);
                }
            }
        }

        // 2. to_pc_reg read (conditional on has_pc_read)
        {
            uint32_t timestamp = record.from_timestamp + 2;
            if (has_pc_read) {
                mem_helper.fill(
                    row.slice_from(COL_INDEX(Cols, to_pc_read_aux)),
                    record.to_pc_read_aux.prev_timestamp,
                    timestamp
                );
            } else {
                mem_helper.fill_zero(row.slice_from(COL_INDEX(Cols, to_pc_read_aux)));
            }
        }

        // 1. to_fp_reg read (conditional on has_fp_read)
        {
            uint32_t timestamp = record.from_timestamp + 1;
            if (has_fp_read) {
                mem_helper.fill(
                    row.slice_from(COL_INDEX(Cols, to_fp_read_aux)),
                    record.to_fp_read_aux.prev_timestamp,
                    timestamp
                );
            } else {
                mem_helper.fill_zero(row.slice_from(COL_INDEX(Cols, to_fp_read_aux)));
            }
        }

        // 0. FP read
        mem_helper.fill(
            row.slice_from(COL_INDEX(Cols, fp_read_aux)),
            record.fp_read_aux.prev_timestamp,
            record.from_timestamp
        );

        // Carry-chain limbs for CALL/CALL_INDIRECT (has_save)
        if (has_save) {
            uint32_t new_fp = fp + to_fp_operand;

            uint16_t offset_lo = (uint16_t)(to_fp_operand & 0xffff);
            uint16_t offset_hi = (uint16_t)(to_fp_operand >> 16);
            uint16_t new_fp_lo = (uint16_t)(new_fp & 0xffff);
            uint16_t new_fp_hi = (uint16_t)(new_fp >> 16);

            uint16_t limbs[2] = { offset_lo, offset_hi };
            COL_WRITE_ARRAY(row, Cols, offset_limbs, limbs);
            uint16_t nfp_limbs[2] = { new_fp_lo, new_fp_hi };
            COL_WRITE_ARRAY(row, Cols, new_fp_limbs, nfp_limbs);

            range_checker.add_count((uint32_t)offset_lo, 16);
            range_checker.add_count((uint32_t)offset_hi, pointer_max_bits - 16);
            range_checker.add_count((uint32_t)new_fp_lo, 16);
            range_checker.add_count((uint32_t)new_fp_hi, pointer_max_bits - 16);
        } else {
            // RET: zero out offset_limbs and new_fp_limbs (not constrained but
            // must be zero so they don't corrupt the trace polynomial).
            uint16_t zeros[2] = { 0, 0 };
            COL_WRITE_ARRAY(row, Cols, offset_limbs, zeros);
            COL_WRITE_ARRAY(row, Cols, new_fp_limbs, zeros);
        }
    }
};
