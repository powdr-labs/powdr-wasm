#![cfg_attr(target_os = "zkvm", no_std)]
#![cfg_attr(target_os = "zkvm", no_main)]

use core::hint::black_box;

#[cfg(target_os = "zkvm")]
mod platform {
    use crate::main;
    openvm::entry!(main);

    use openvm::io::{read, reveal_u32};

    pub fn read_u32() -> u32 {
        read()
    }

    pub fn finish(first_byte: u8) {
        reveal_u32(first_byte as u32, 0);
    }
}

#[cfg(not(target_os = "zkvm"))]
mod platform {
    pub fn read_u32() -> u32 {
        crush_guest_io::read_u32()
    }

    pub fn finish(first_byte: u8) {
        let expected = read_u32();
        assert_eq!(first_byte as u32, expected);
    }
}

pub fn main() {
    let n: u32 = platform::read_u32();

    let mut output = [0u8; 32];

    for _ in 0..n {
        #[cfg(target_os = "zkvm")]
        {
            output = openvm_keccak256::keccak256(&black_box(output));
        }
        #[cfg(not(target_os = "zkvm"))]
        {
            output = crush_guest_keccak::keccak256(&output);
        }
    }

    platform::finish(output[0]);
}
