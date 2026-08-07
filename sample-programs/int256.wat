;; Minimal end-to-end check of the Int256 precompiles.
;;
;; Each intrinsic takes three wasm pointers -- destination, then the two operands -- to
;; 32-byte little-endian integers. The chips read those pointers from registers and the
;; data from the heap.
;;
;; Operands are chosen to exercise carries and truncation rather than trivial arithmetic:
;;
;;   a = 2^200 + 2^64 - 1     b = 2^200 + 1
;;   a + b = 2^201 + 2^64     (carry propagates out of the low limbs)
;;   a * b = 2^64 - 1         (mod 2^256; every high limb is truncated away)
;;
;; Requires --int256. Returns 0 on success; traps (unreachable) on mismatch.
;;
;; Build:  wat2wasm sample-programs/int256.wat -o sample-programs/int256.wasm
;; Run:    cargo run -r -- run --int256 sample-programs/int256.wasm int256_add
;;         cargo run -r -- run --int256 sample-programs/int256.wasm int256_mul

(import "env" "__int256_add" (func $int256_add (param i32) (param i32) (param i32)))
(import "env" "__int256_mul" (func $int256_mul (param i32) (param i32) (param i32)))

(memory 1)

;; All buffers 4-byte aligned and 32 bytes apart.
(global $a i32 (i32.const 64))
(global $b i32 (i32.const 96))
(global $out i32 (i32.const 128))
(global $expected i32 (i32.const 160))

;; a = 2^200 + 2^64 - 1
(data (i32.const 64)
    "\ff\ff\ff\ff\ff\ff\ff\ff\00\00\00\00\00\00\00\00"
    "\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00")
;; b = 2^200 + 1
(data (i32.const 96)
    "\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00")

;; a + b = 2^201 + 2^64, staged at 192 and copied into `expected` by the test.
(data (i32.const 192)
    "\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00"
    "\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00")
;; a * b mod 2^256 = 2^64 - 1
(data (i32.const 224)
    "\ff\ff\ff\ff\ff\ff\ff\ff\00\00\00\00\00\00\00\00"
    "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00")

;; Trap unless the 32 bytes at $out equal the 32 bytes at `want`.
(func $expect (param $want i32)
    (local $i i32)
    block $done
        loop $next
            local.get $i
            i32.const 32
            i32.ge_u
            br_if $done
            (i32.load8_u (i32.add (global.get $out) (local.get $i)))
            (i32.load8_u (i32.add (local.get $want) (local.get $i)))
            i32.ne
            if
                unreachable
            end
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            br $next
        end
    end
)

(func (export "int256_add") (result i32)
    (call $int256_add (global.get $out) (global.get $a) (global.get $b))
    (call $expect (i32.const 192))
    i32.const 0
)

(func (export "int256_mul") (result i32)
    (call $int256_mul (global.get $out) (global.get $a) (global.get $b))
    (call $expect (i32.const 224))
    i32.const 0
)
