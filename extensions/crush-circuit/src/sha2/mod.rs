// The #[create_handler] macro generates unsafe fn bodies that call unsafe functions without
// inner unsafe blocks. Upstream's sha2 crate is edition 2021, where that is allowed;
// crush-circuit is edition 2024, where it warns. Scoped to this module.
#![allow(unsafe_op_in_unsafe_fn)]

//! SHA-2 support, ported from OpenVM's `openvm-sha2-circuit` (v2.0.0-beta.2).
//!
//! One opcode per digest size (`SHA256`, `SHA512`), each performing a single
//! compression: it reads the previous hash state and one message block and writes
//! the new state. Padding and the block loop live in the guest, not in any AIR.
//!
//! Two chips per config, joined by a bus: a main chip facing the VM (register reads,
//! memory) and a block-hasher chip wrapping the `openvm-sha2-air` sub-AIR. The
//! sub-AIR is depended on rather than forked; the only crush-specific change is that
//! the main chip reads registers relative to the frame pointer.

mod block_hasher_chip;
mod config;
mod execution;
mod main_chip;
mod trace;

mod extension;
pub use extension::*;

use std::marker::PhantomData;

pub use block_hasher_chip::*;
pub use main_chip::*;
pub use trace::*;

#[derive(derive_new::new, Clone)]
pub struct Sha2VmExecutor<C: Sha2Config> {
    pub offset: usize,
    pub pointer_max_bits: usize,
    _phantom: PhantomData<C>,
}

// Indicates the message type of the interactions on the sha bus
#[repr(u8)]
pub enum MessageType {
    State,
    Message1,
    Message2,
}
