;; Minimal end-to-end check of the XORIN precompile.
;;
;; XORs an input block into a sponge buffer via the `env.__xorin` import
;; and verifies the result, trapping on mismatch. Uses uniform fill bytes so the
;; expected value is a constant and no lookup table is needed:
;;
;;   0xa5 ^ 0x3c = 0x99
;;
;; Two exports:
;;
;;   xorin_full_block    len = 136 (the keccak rate): every one of the 34 words
;;                       is absorbed.
;;   xorin_partial_block len = 12: only the first 12 bytes are absorbed and the
;;                       rest of the rate block must be left untouched, which is
;;                       what the AIR's is_padding_bytes columns are for. No
;;                       guest reaches this path -- the Rust guest sponge always
;;                       submits a full block -- so it is only exercised here.
;;
;; Requires --keccak. Returns 0 on success; traps (unreachable) on mismatch.
;;
;; Build:  wat2wasm sample-programs/xorin.wat -o sample-programs/xorin.wasm
;; Run:    cargo run -r -- run --keccak sample-programs/xorin.wasm xorin_full_block
;;         cargo run -r -- run --keccak sample-programs/xorin.wasm xorin_partial_block

(import "env" "__xorin"
    (func $xorin (param i32) (param i32) (param i32)))

(memory 1)

;; Both 4-byte aligned, as the precompile requires. 136 bytes each, no overlap.
(global $buffer i32 (i32.const 64))
(global $input i32 (i32.const 512))

(global $rate i32 (i32.const 136))
(global $buf_fill i32 (i32.const 0xa5))
(global $in_fill i32 (i32.const 0x3c))

;; Fill `len` bytes at `ptr` with `byte`.
(func $fill (param $ptr i32) (param $byte i32) (param $len i32)
    (local $i i32)
    block $done
        loop $next
            local.get $i
            local.get $len
            i32.ge_u
            br_if $done
            (i32.store8 (i32.add (local.get $ptr) (local.get $i)) (local.get $byte))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            br $next
        end
    end
)

;; Trap unless the `len` bytes at `ptr` all equal `byte`.
(func $expect_fill (param $ptr i32) (param $byte i32) (param $len i32)
    (local $i i32)
    block $done
        loop $next
            local.get $i
            local.get $len
            i32.ge_u
            br_if $done
            (i32.load8_u (i32.add (local.get $ptr) (local.get $i)))
            local.get $byte
            i32.ne
            if
                unreachable
            end
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            br $next
        end
    end
)

;; XOR `len` bytes in, then check the absorbed prefix and the untouched tail.
(func $run (param $len i32)
    (call $fill (global.get $buffer) (global.get $buf_fill) (global.get $rate))
    (call $fill (global.get $input) (global.get $in_fill) (global.get $rate))

    (call $xorin (global.get $buffer) (global.get $input) (local.get $len))

    ;; absorbed prefix: 0xa5 ^ 0x3c
    (call $expect_fill (global.get $buffer) (i32.const 0x99) (local.get $len))
    ;; untouched tail: still 0xa5
    (call $expect_fill
        (i32.add (global.get $buffer) (local.get $len))
        (global.get $buf_fill)
        (i32.sub (global.get $rate) (local.get $len)))
)

(func (export "xorin_full_block") (result i32)
    (call $run (global.get $rate))
    i32.const 0
)

(func (export "xorin_partial_block") (result i32)
    (call $run (i32.const 12))
    i32.const 0
)
