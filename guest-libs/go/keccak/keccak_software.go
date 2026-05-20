//go:build !crush_keccak

package keccak

import "golang.org/x/crypto/sha3"

// Keccak256 computes the keccak256 hash of data using software implementation.
func Keccak256(data []byte) [32]byte {
	h := sha3.NewLegacyKeccak256()
	h.Write(data)
	var out [32]byte
	h.Sum(out[:0])
	return out
}
