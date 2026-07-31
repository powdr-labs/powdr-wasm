;; Minimal end-to-end check of the KECCAKF precompile.
;;
;; Applies one keccak-f[1600] permutation to the all-zero state via the
;; `env.__native_keccakf` import, then compares all 200 result bytes against the
;; known Keccak-f[1600] output for a zero input state, trapping on any mismatch.
;; The first lane, 0xf1258f7940e1dde7, is the standard XKCP test vector.
;;
;; Requires --keccak. Returns 0 on success; traps (unreachable) on mismatch.
;;
;; Build:  wat2wasm sample-programs/keccakf.wat -o sample-programs/keccakf.wasm
;; Run:    cargo run -r -- run --keccak sample-programs/keccakf.wasm keccakf_zero_state

(import "env" "__native_keccakf" (func $native_keccakf (param i32)))

(memory 1)

;; The 200-byte state lives at 64 (4-byte aligned, as the precompile requires)
;; and starts all-zero, which is the memory default.
(global $state i32 (i32.const 64))

;; Expected keccak-f output for the all-zero state, at 512.
(data (i32.const 512)
    "\e7\dd\e1\40\79\8f\25\f1\8a\47\c0\33\f9\cc\d5\84\ee\a9\5a\a6"
    "\1e\26\98\d5\4d\49\80\6f\30\47\15\bd\57\d0\53\62\05\4e\28\8b"
    "\d4\6f\8e\7f\2d\a4\97\ff\c4\47\46\a4\a0\e5\fe\90\76\2e\19\d6"
    "\0c\da\5b\8c\9c\05\19\1b\f7\a6\30\ad\64\fc\8f\d0\b7\5a\93\30"
    "\35\d6\17\23\3f\a9\5a\eb\03\21\71\0d\26\e6\a6\a9\5f\55\cf\db"
    "\16\7c\a5\81\26\c8\47\03\cd\31\b8\43\9f\56\a5\11\1a\2f\f2\01"
    "\61\ae\d9\21\5a\63\e5\05\f2\70\c9\8c\f2\fe\be\64\11\66\c4\7b"
    "\95\70\36\61\cb\0e\d0\4f\55\5a\7c\b8\c8\32\cf\1c\8a\e8\3e\8c"
    "\14\26\3a\ae\22\79\0c\94\e4\09\c5\a2\24\f9\41\18\c2\65\04\e7"
    "\26\35\f5\16\3b\a1\30\7f\e9\44\f6\75\49\a2\ec\5c\7b\ff\f1\ea"
)

(func (export "keccakf_zero_state") (result i32)
    (local $i i32)

    ;; Permute the zero state in place.
    (call $native_keccakf (global.get $state))

    ;; Compare all 200 bytes against the expected output.
    block $done
        loop $cmp
            local.get $i
            i32.const 200
            i32.ge_u
            br_if $done

            ;; state[i] != expected[i]  ->  trap
            (i32.load8_u (i32.add (global.get $state) (local.get $i)))
            (i32.load8_u (i32.add (i32.const 512) (local.get $i)))
            i32.ne
            if
                unreachable
            end

            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            br $cmp
        end
    end

    i32.const 0
)
