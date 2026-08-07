;; Minimal end-to-end check of the Fp2 arithmetic precompiles.
;;
;; Fp2 is the quadratic extension Fp[u]/(u^2 + 1), so an element is two 32-byte
;; little-endian coefficients laid out back to back, c0 then c1. Each intrinsic takes three
;; wasm pointers -- destination, then the two operands -- to such 64-byte values.
;;
;; The index in the name selects which configured modulus to use, so this program must be run
;; with bn254 first in the --fp2 list. Setup checks the modulus it is handed against the one
;; the chip was built with, which is what makes that correspondence a checked one rather than
;; a convention. Setup only reads the first 32 bytes it is pointed at; the rest of the operand
;; and the whole second operand are read but unconstrained, so they only have to be readable.
;;
;;   p = the BN254 coordinate prime
;;   a = (p-3) + 7u          b = 5 + (p-2)u
;;   a + b = 2 + 5u          a - b = (p-8) + 9u
;;   a * b = (p-2) + 41u     a / b = (p-2) + u
;;
;; Returns 0 on success; traps (unreachable) on mismatch.
;;
;; Build:  wat2wasm sample-programs/fp2.wat -o sample-programs/fp2.wasm
;; Run:    cargo run -r -- run --fp2 bn254 sample-programs/fp2.wasm fp2_addsub
;;         cargo run -r -- run --fp2 bn254 sample-programs/fp2.wasm fp2_muldiv

(import "env" "__fp2_0_setup_addsub" (func $setup_addsub (param i32) (param i32) (param i32)))
(import "env" "__fp2_0_setup_muldiv" (func $setup_muldiv (param i32) (param i32) (param i32)))
(import "env" "__fp2_0_add" (func $add (param i32) (param i32) (param i32)))
(import "env" "__fp2_0_sub" (func $sub (param i32) (param i32) (param i32)))
(import "env" "__fp2_0_mul" (func $mul (param i32) (param i32) (param i32)))
(import "env" "__fp2_0_div" (func $div (param i32) (param i32) (param i32)))

(memory 1)

;; Fp2 values are 64 bytes, so the buffers are 64 bytes apart.
(global $a i32 (i32.const 64))
(global $b i32 (i32.const 128))
(global $out i32 (i32.const 192))
;; Setup's first operand: p in the low 32 bytes, anything after.
(global $setup i32 (i32.const 256))

;; a = (p-3) + 7u
(data (i32.const 64)
    "\44\fd\7c\d8\16\8c\20\3c\8d\ca\71\68\91\6a\81\97"
    "\5d\58\81\81\b6\45\50\b8\29\a0\31\e1\72\4e\64\30"
    "\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00")
;; b = 5 + (p-2)u
(data (i32.const 128)
    "\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\45\fd\7c\d8\16\8c\20\3c\8d\ca\71\68\91\6a\81\97"
    "\5d\58\81\81\b6\45\50\b8\29\a0\31\e1\72\4e\64\30")
;; p, followed by 32 zero bytes so the whole 64-byte read is in bounds.
(data (i32.const 256)
    "\47\fd\7c\d8\16\8c\20\3c\8d\ca\71\68\91\6a\81\97"
    "\5d\58\81\81\b6\45\50\b8\29\a0\31\e1\72\4e\64\30")

;; Expected results.
;; a + b = 2 + 5u
(data (i32.const 512)
    "\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00")
;; a - b = (p-8) + 9u
(data (i32.const 576)
    "\3f\fd\7c\d8\16\8c\20\3c\8d\ca\71\68\91\6a\81\97"
    "\5d\58\81\81\b6\45\50\b8\29\a0\31\e1\72\4e\64\30"
    "\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00")
;; a * b = (p-2) + 41u
(data (i32.const 640)
    "\46\fd\7c\d8\16\8c\20\3c\8d\ca\71\68\91\6a\81\97"
    "\5d\58\81\81\b6\45\50\b8\29\a0\31\e1\72\4e\64\30"
    "\29\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00")
;; a / b = (p-2) + u
(data (i32.const 704)
    "\46\fd\7c\d8\16\8c\20\3c\8d\ca\71\68\91\6a\81\97"
    "\5d\58\81\81\b6\45\50\b8\29\a0\31\e1\72\4e\64\30"
    "\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00")

;; Trap unless the 64 bytes at $out equal the 64 bytes at `want`.
(func $expect (param $want i32)
    (local $i i32)
    block $done
        loop $next
            local.get $i
            i32.const 64
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

(func (export "fp2_addsub") (result i32)
    ;; Setup writes a result of its own into $out, which the checks below overwrite.
    (call $setup_addsub (global.get $out) (global.get $setup) (global.get $setup))
    (call $add (global.get $out) (global.get $a) (global.get $b))
    (call $expect (i32.const 512))
    (call $sub (global.get $out) (global.get $a) (global.get $b))
    (call $expect (i32.const 576))
    i32.const 0
)

(func (export "fp2_muldiv") (result i32)
    (call $setup_muldiv (global.get $out) (global.get $setup) (global.get $setup))
    (call $mul (global.get $out) (global.get $a) (global.get $b))
    (call $expect (i32.const 640))
    (call $div (global.get $out) (global.get $a) (global.get $b))
    (call $expect (i32.const 704))
    i32.const 0
)
