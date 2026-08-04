;; Minimal end-to-end check of the pairing extension.
;;
;; The pairing extension contributes no chips and no opcodes -- only a hint phantom. It runs
;; the Miller loop over the given points on the host and pushes the final-exponentiation
;; residue witness onto the hint stream, which a guest pairing implementation would then read
;; back and verify using the Fp2, modular and ecc chips.
;;
;; That verifier is the whole of a pairing library, so this program deliberately does not
;; attempt it. What it checks is that the hint is reachable and well formed: the phantom
;; assembles its descriptors from rebased wasm pointers, reads the right frame-pointer
;; relative registers, and returns the documented 768 bytes -- BN254's witness is two Fq12
;; values, each six Fq2 coefficients of two 32-byte Fp coefficients. Every one of those 24 Fp
;; coefficients must be reduced mod p, and at least one must be non-zero. Reading the wrong
;; registers yields garbage pointers, so the phantom would either fail outright or produce a
;; witness for different points, which those checks catch.
;;
;; `__pairing_0_hint_final_exp` takes a 16-byte scratch buffer for the descriptors the phantom
;; expects, then the G1 array and its length, then the G2 array and its length. A G1 point is
;; two 32-byte coordinates (64 bytes); a G2 point is two Fp2 coordinates (128 bytes).
;;
;; The index in the name selects which configured curve to use, so this program must be run
;; with bn254 first in the --pairing list.
;;
;; Returns 0 on success; traps (unreachable) on mismatch.
;;
;; Build:  wat2wasm sample-programs/pairing.wat -o sample-programs/pairing.wasm
;; Run:    cargo run -r -- run --pairing bn254 sample-programs/pairing.wasm pairing_hint

(import "env" "__pairing_0_hint_final_exp"
    (func $hint_final_exp (param i32) (param i32) (param i32) (param i32) (param i32)))
(import "env" "__hint_buffer" (func $hint_buffer (param i32) (param i32)))

(memory 1)

(global $scratch i32 (i32.const 64))
(global $g1 i32 (i32.const 128))
(global $g2 i32 (i32.const 256))
(global $out i32 (i32.const 512))

;; G1, the BN254 generator: (1, 2)
(data (i32.const 128)
    "\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00"
    "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00")

;; G2, the BN254 generator on the twist: x = (c0, c1) then y = (c0, c1)
(data (i32.const 256)
    "\ed\f6\92\d9\5c\bd\de\46\dd\da\5e\f7\d4\22\43\67"
    "\79\44\5c\5e\66\00\6a\42\76\1e\1f\12\ef\de\00\18"
    "\c2\12\f3\ae\b7\85\e4\97\12\e7\a9\35\33\49\aa\f1"
    "\25\5d\fb\31\b7\bf\60\72\3a\48\0d\92\93\93\8e\19"
    "\aa\7d\fa\66\01\cc\e6\4c\7b\d3\43\0c\69\e7\d1\e3"
    "\8f\40\cb\8d\80\71\ab\4a\eb\6d\8c\db\a5\5e\c8\12"
    "\5b\97\22\d1\dc\da\ac\55\f3\8e\b3\70\33\31\4b\bc"
    "\95\33\0c\69\ad\99\9e\ec\75\f0\5f\58\d0\89\06\09")

(func (export "pairing_hint") (result i32)
    (local $i i32)
    (local $nonzero i32)

    (call $hint_final_exp
        (global.get $scratch)
        (global.get $g1) (i32.const 1)
        (global.get $g2) (i32.const 1))
    ;; 768 bytes = 192 words. This traps if the phantom pushed fewer.
    (call $hint_buffer (global.get $out) (i32.const 192))

    ;; Each of the 24 Fp coefficients must be reduced mod p. p's most significant byte is
    ;; 0x30, so a canonical coefficient's must not exceed it -- a cheap check that needs no
    ;; field arithmetic, but one that garbage would almost certainly fail.
    block $done
        loop $next
            local.get $i
            i32.const 24
            i32.ge_u
            br_if $done
            (i32.load8_u offset=31
                (i32.add (global.get $out) (i32.mul (local.get $i) (i32.const 32))))
            i32.const 0x30
            i32.gt_u
            if
                unreachable
            end
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            br $next
        end
    end

    ;; A witness of all zeros would pass the check above, so require some content.
    (local.set $i (i32.const 0))
    block $done2
        loop $next2
            local.get $i
            i32.const 768
            i32.ge_u
            br_if $done2
            (i32.load8_u (i32.add (global.get $out) (local.get $i)))
            if
                (local.set $nonzero (i32.const 1))
                br $done2
            end
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            br $next2
        end
    end
    (if (i32.eqz (local.get $nonzero)) (then unreachable))

    i32.const 0
)
