module github.com/powdr-labs/go-keccak-guest

go 1.24

require github.com/powdr-labs/powdr-wasm/guest-keccak-go v0.0.0

require (
	golang.org/x/crypto v0.32.0 // indirect
	golang.org/x/sys v0.29.0 // indirect
)

replace github.com/powdr-labs/powdr-wasm/guest-keccak-go => ../../guest-libs/go/keccak
