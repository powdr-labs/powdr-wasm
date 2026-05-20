//go:build wasip1

package main

import (
	"fmt"

	keccak "github.com/powdr-labs/powdr-wasm/guest-keccak-go"
)

func main() {
	// Read number of iterations
	n := readU32()

	// Read expected first byte of output (for verification)
	expected := readU32()

	// Iterative keccak256: hash the output repeatedly
	var output [32]byte
	for i := uint32(0); i < n; i++ {
		output = keccak.Keccak256(output[:])
	}

	if uint32(output[0]) != expected {
		panic(fmt.Sprintf("MISMATCH: got %d, expected %d", output[0], expected))
	}
}
