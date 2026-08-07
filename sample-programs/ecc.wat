;; Minimal end-to-end check of the Weierstrass elliptic curve precompiles.
;;
;; A point is its two affine coordinates, each a 32-byte little-endian field element, laid out
;; back to back -- 64 bytes in all. `add_ne` takes three wasm pointers (destination, then the
;; two summands) and `double` takes two; the chips read the pointers from registers and the
;; coordinates from the heap.
;;
;; The index in the name selects which configured curve to use, so this program must be run
;; with secp256k1 first in the --ecc list. Setup checks what it is handed against the curve
;; parameters the chip was built with, which is what makes that correspondence a checked one
;; rather than a convention: `setup_add_ne` checks the coordinate modulus, and
;; `setup_double` checks the modulus followed by the curve coefficient `a`.
;;
;; `add_ne` is addition for distinct points only -- it has no case for P = Q -- so the two
;; operations are checked against secp256k1's generator this way:
;;
;;   3G = add_ne(2G, G)      4G = double(2G)
;;
;; Returns 0 on success; traps (unreachable) on mismatch.
;;
;; Build:  wat2wasm sample-programs/ecc.wat -o sample-programs/ecc.wasm
;; Run:    cargo run -r -- run --ecc secp256k1 sample-programs/ecc.wasm ecc_add_ne
;;         cargo run -r -- run --ecc secp256k1 sample-programs/ecc.wasm ecc_double

(import "env" "__ecc_0_setup_add_ne" (func $setup_add_ne (param i32) (param i32) (param i32)))
(import "env" "__ecc_0_setup_double" (func $setup_double (param i32) (param i32) (param i32)))
(import "env" "__ecc_0_add_ne" (func $add_ne (param i32) (param i32) (param i32)))
(import "env" "__ecc_0_double" (func $double (param i32) (param i32) (param i32)))

(memory 1)

;; Points are 64 bytes, so the buffers are 64 bytes apart.
(global $g i32 (i32.const 64))
(global $g2 i32 (i32.const 128))
(global $out i32 (i32.const 192))
;; Setup's first operand: the coordinate modulus p, then the curve coefficient a = 0. The
;; doubling setup checks both; the addition setup only checks p.
(global $setup i32 (i32.const 256))

;; G, the secp256k1 generator
(data (i32.const 64)
    "\98\17\f8\16\5b\81\f2\59\d9\28\ce\2d\db\fc\9b\02"
    "\07\0b\87\ce\95\62\a0\55\ac\bb\dc\f9\7e\66\be\79"
    "\b8\d4\10\fb\8f\d0\47\9c\19\54\85\a6\48\b4\17\fd"
    "\a8\08\11\0e\fc\fb\a4\5d\65\c4\a3\26\77\da\3a\48")
;; 2G
(data (i32.const 128)
    "\e5\9e\70\5c\b9\09\ac\ab\a7\3c\ef\8c\4b\8e\77\5c"
    "\d8\7c\c0\95\6e\40\45\30\6d\7d\ed\41\94\7f\04\c6"
    "\2a\e5\cf\50\a9\31\64\23\e1\d0\66\32\65\32\f6\f7"
    "\ee\ea\6c\46\19\84\c5\a3\39\c3\3d\a6\fe\68\e1\1a")
;; p, then a = 0
(data (i32.const 256)
    "\2f\fc\ff\ff\fe\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff"
    "\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff\ff")

;; Expected results.
;; 3G
(data (i32.const 512)
    "\f9\36\e0\bc\13\f1\01\86\b0\99\6f\83\45\c8\31\b5"
    "\29\52\9d\f8\85\4f\34\49\10\c3\58\92\01\8a\30\f9"
    "\72\e6\b8\84\75\fd\b9\6c\1b\23\c2\34\99\a9\00\65"
    "\56\f3\37\2a\e6\37\e3\0f\14\e8\2d\63\0f\7b\8f\38")
;; 4G
(data (i32.const 576)
    "\13\cd\c4\e8\ab\94\fa\74\84\75\e0\0e\90\13\6c\cc"
    "\04\14\0b\93\04\49\1e\58\f3\80\0d\c1\f1\db\93\e4"
    "\22\99\73\47\dc\7b\e9\cf\40\fe\bd\bf\33\ae\67\d9"
    "\48\14\a5\8e\09\e2\42\56\b7\55\d4\a0\3e\99\ed\51")

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

(func (export "ecc_add_ne") (result i32)
    ;; Setup writes a result of its own into $out, which the check below overwrites. Its third
    ;; operand is read but unconstrained; $g is a convenient readable 64 bytes.
    (call $setup_add_ne (global.get $out) (global.get $setup) (global.get $g))
    (call $add_ne (global.get $out) (global.get $g2) (global.get $g))
    (call $expect (i32.const 512))
    i32.const 0
)

(func (export "ecc_double") (result i32)
    ;; The doubling chip reads a single operand, so its third argument is ignored: the
    ;; translation forces the encoded operand to zero and the adapter never dereferences it.
    (call $setup_double (global.get $out) (global.get $setup) (global.get $g))
    (call $double (global.get $out) (global.get $g2) (global.get $g))
    (call $expect (i32.const 576))
    i32.const 0
)
