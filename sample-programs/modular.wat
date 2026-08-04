;; Minimal end-to-end check of the modular arithmetic precompiles.
;;
;; Each intrinsic takes three wasm pointers -- destination, then the two operands -- to
;; 32-byte little-endian field elements. The chips read the pointers from registers and the
;; data from the heap.
;;
;; The index in the name selects which configured modulus to use, so this program must be run
;; with secp256k1 first in the --modular list. Each chip needs its SETUP instruction before
;; its first real operation; setup checks the modulus it is handed against the one the chip
;; was built with, which is what makes that correspondence a checked one rather than a
;; convention.
;;
;; Setup reads a second operand it does not check against anything, because it goes through
;; the same adapter as the real operation. It still has to be a reduced field element: the
;; equality chip range checks its second operand whether or not it is setting up. `b` below
;; serves that purpose.
;;
;; Operands are chosen so every result needs a reduction:
;;
;;   p = 2^256 - 2^32 - 977    a = p - 3    b = 5
;;   a + b = 2 (mod p)         a - b = p - 8
;;   a * b = p - 15 (mod p)    a / b = (p-3) * b^-1 (mod p)
;;
;; Returns 0 on success; traps (unreachable) on mismatch.
;;
;; Build:  wat2wasm sample-programs/modular.wat -o sample-programs/modular.wasm
;; Run:    cargo run -r -- run --modular secp256k1 sample-programs/modular.wasm modular_addsub
;;         cargo run -r -- run --modular secp256k1 sample-programs/modular.wasm modular_muldiv
;;         cargo run -r -- run --modular secp256k1 sample-programs/modular.wasm modular_is_eq

(import "env" "__modular_0_setup_addsub" (func $setup_addsub (param i32) (param i32) (param i32)))
(import "env" "__modular_0_setup_muldiv" (func $setup_muldiv (param i32) (param i32) (param i32)))
(import "env" "__modular_0_setup_iseq" (func $setup_iseq (param i32) (param i32) (result i32)))
(import "env" "__modular_0_add" (func $add (param i32) (param i32) (param i32)))
(import "env" "__modular_0_sub" (func $sub (param i32) (param i32) (param i32)))
(import "env" "__modular_0_mul" (func $mul (param i32) (param i32) (param i32)))
(import "env" "__modular_0_div" (func $div (param i32) (param i32) (param i32)))
(import "env" "__modular_0_is_eq" (func $is_eq (param i32) (param i32) (result i32)))

(memory 1)

;; All buffers 4-byte aligned and 32 bytes apart.
(global $p i32 (i32.const 64))
(global $a i32 (i32.const 96))
(global $b i32 (i32.const 128))
(global $out i32 (i32.const 160))

;; p, the secp256k1 coordinate prime. Also the operand every setup instruction is checked
;; against.
(data (i32.const 64)
    "\2f\fc\ff\ff\fe\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff"
    "\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff")
;; a = p - 3
(data (i32.const 96)
    "\2c\fc\ff\ff\fe\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff"
    "\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff")
;; b = 5
(data (i32.const 128)
    "\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00")

;; Expected results, staged out of the way of the operands.
;; a + b = 2
(data (i32.const 256)
    "\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00")
;; a - b = p - 8
(data (i32.const 288)
    "\27\fc\ff\ff\fe\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff"
    "\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff")
;; a * b = p - 15
(data (i32.const 320)
    "\20\fc\ff\ff\fe\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff"
    "\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff")
;; a / b
(data (i32.const 352)
    "\3c\ff\ff\ff\32\33\33\33\33\33\33\33\33\33\33\33"
    "\33\33\33\33\33\33\33\33\33\33\33\33\33\33\33\33")

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

(func (export "modular_addsub") (result i32)
    ;; Setup writes a result of its own into $out, which the checks below overwrite.
    (call $setup_addsub (global.get $out) (global.get $p) (global.get $b))
    (call $add (global.get $out) (global.get $a) (global.get $b))
    (call $expect (i32.const 256))
    (call $sub (global.get $out) (global.get $a) (global.get $b))
    (call $expect (i32.const 288))
    i32.const 0
)

(func (export "modular_muldiv") (result i32)
    (call $setup_muldiv (global.get $out) (global.get $p) (global.get $b))
    (call $mul (global.get $out) (global.get $a) (global.get $b))
    (call $expect (i32.const 320))
    (call $div (global.get $out) (global.get $a) (global.get $b))
    (call $expect (i32.const 352))
    i32.const 0
)

(func (export "modular_is_eq") (result i32)
    ;; Unlike the others, equality returns its result in a register.
    (drop (call $setup_iseq (global.get $p) (global.get $b)))
    (if (i32.eqz (call $is_eq (global.get $a) (global.get $a))) (then unreachable))
    (if (call $is_eq (global.get $a) (global.get $b)) (then unreachable))
    i32.const 0
)
