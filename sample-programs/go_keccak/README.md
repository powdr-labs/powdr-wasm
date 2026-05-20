# Go Keccak Guest

Iterative keccak256 benchmark as a Go WASI guest. Two build variants:

- **Software**: Uses `golang.org/x/crypto/sha3.NewLegacyKeccak256()` (same as geth)
- **Precompile**: Uses `__native_keccak256` syscall via the crush keccak extension

## Build

```bash
cd sample-programs/go_keccak

# Software keccak
GOOS=wasip1 GOARCH=wasm go build -gcflags='all=-d=softfloat' -o ../go_keccak_software.wasm .

# Precompile keccak (requires --keccak flag when running)
GOOS=wasip1 GOARCH=wasm go build -gcflags='all=-d=softfloat' -tags crush_keccak -o ../go_keccak_precompile.wasm .
```

## Run

```bash
# Software (1000 iterations, expected first byte = 39)
cargo run -r -- run --unaligned-memory sample-programs/go_keccak_software.wasm _start --input 1000 --input 39

# Precompile
cargo run -r -- run --keccak --unaligned-memory sample-programs/go_keccak_precompile.wasm _start --input 1000 --input 39
```

## Prove

```bash
# Software
cargo run -r -- compile --unaligned-memory --apc-count 0 --output-dir /tmp/go_keccak_sw/compiled \
  sample-programs/go_keccak_software.wasm _start --input 1000 --input 39
cargo run -r -- prove --compiled-dir /tmp/go_keccak_sw/compiled --recursion \
  --input 1000 --input 39 --metrics /tmp/go_keccak_sw/metrics.json

# Precompile
cargo run -r -- compile --keccak --unaligned-memory --apc-count 0 --output-dir /tmp/go_keccak_hw/compiled \
  sample-programs/go_keccak_precompile.wasm _start --input 1000 --input 39
cargo run -r -- prove --compiled-dir /tmp/go_keccak_hw/compiled --recursion \
  --input 1000 --input 39 --metrics /tmp/go_keccak_hw/metrics.json
```
