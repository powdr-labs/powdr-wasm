// Package keccak provides a keccak256 implementation that uses the crush
// __native_keccak256 precompile when compiled with the crush_keccak build tag,
// and falls back to golang.org/x/crypto/sha3 otherwise.
//
// Usage:
//
//	import "github.com/powdr-labs/powdr-wasm/guest-keccak-go"
//	hash := keccak.Keccak256(data)
package keccak
