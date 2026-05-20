//go:build crush_keccak

package keccak

import "unsafe"

//go:wasmimport env __native_keccak256
func nativeKeccak256(input unsafe.Pointer, inputLen uint32, output unsafe.Pointer)

// Keccak256 computes the keccak256 hash of data using the crush precompile.
func Keccak256(data []byte) [32]byte {
	var out [32]byte
	if len(data) == 0 {
		nativeKeccak256(unsafe.Pointer(&out[0]), 0, unsafe.Pointer(&out[0]))
	} else {
		nativeKeccak256(unsafe.Pointer(&data[0]), uint32(len(data)), unsafe.Pointer(&out[0]))
	}
	return out
}
