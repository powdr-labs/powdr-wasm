//go:build wasip1

package main

import "unsafe"

// WOMIR guest-io imports (env module).
// Protocol: __hint_input prepares next item, __hint_buffer reads words.
// Each item: [byte_len_u32_le, ...data_words_padded_to_4bytes]

//go:wasmimport env __hint_input
func hintInput()

//go:wasmimport env __hint_buffer
func hintBuffer(ptr unsafe.Pointer, numWords uint32)

//go:wasmimport env __debug_print
func debugPrint(ptr unsafe.Pointer, numBytes uint32)

func readWord() uint32 {
	var buf [4]byte
	hintBuffer(unsafe.Pointer(&buf[0]), 1)
	return uint32(buf[0]) | uint32(buf[1])<<8 | uint32(buf[2])<<16 | uint32(buf[3])<<24
}

func readU32() uint32 {
	hintInput()
	byteLen := readWord()
	if byteLen != 4 {
		panic("readU32: expected 4-byte item")
	}
	return readWord()
}

func guestPrint(msg string) {
	if len(msg) > 0 {
		debugPrint(unsafe.Pointer(unsafe.StringData(msg)), uint32(len(msg)))
	}
}
