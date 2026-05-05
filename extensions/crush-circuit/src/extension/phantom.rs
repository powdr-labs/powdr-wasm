use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use eyre::bail;
use openvm_circuit::{
    arch::{PhantomSubExecutor, Streams},
    system::memory::online::GuestMemory,
};
use openvm_instructions::PhantomDiscriminant;
use openvm_stark_backend::p3_field::{Field, PrimeField32};
use rand::{Rng, rngs::StdRng};

use crate::adapters::{memory_read, read_rv32_register};
use crate::memory_config::FpMemory;

use openvm_instructions::riscv::RV32_MEMORY_AS;

/// Reads a 32-bit register value from memory, applying FP offset.
fn read_register<F: PrimeField32>(memory: &GuestMemory, reg_offset: u32) -> u32 {
    let fp = memory.fp::<F>();
    read_rv32_register(memory, fp + reg_offset)
}

/// HintInput: Prepares the next input for hinting.
/// Pops from input_stream, prepends 4-byte length, extends hint_stream.
pub struct HintInputSubEx;

impl<F: Field> PhantomSubExecutor<F> for HintInputSubEx {
    fn phantom_execute(
        &self,
        _: &GuestMemory,
        streams: &mut Streams<F>,
        _: &mut StdRng,
        _: PhantomDiscriminant,
        _: u32,
        _: u32,
        _: u16,
    ) -> eyre::Result<()> {
        let mut hint = match streams.input_stream.pop_front() {
            Some(hint) => hint,
            None => {
                bail!("EndOfInputStream");
            }
        };
        streams.hint_stream.clear();
        // Prepend 4-byte LE length
        streams.hint_stream.extend(
            (hint.len() as u32)
                .to_le_bytes()
                .iter()
                .map(|b| F::from_u8(*b)),
        );
        // Pad to 4-byte alignment
        let capacity = hint.len().div_ceil(4) * 4;
        hint.resize(capacity, F::ZERO);
        streams.hint_stream.extend(hint);
        Ok(())
    }
}

/// PrintStr: Reads a string from memory and prints it.
/// a = mem_ptr register offset, b = num_bytes register offset, c_upper = mem_start
pub struct PrintStrSubEx;

impl<F: PrimeField32> PhantomSubExecutor<F> for PrintStrSubEx {
    fn phantom_execute(
        &self,
        memory: &GuestMemory,
        _: &mut Streams<F>,
        _: &mut StdRng,
        _: PhantomDiscriminant,
        a: u32,
        b: u32,
        c_upper: u16,
    ) -> eyre::Result<()> {
        let mem_start = c_upper as u32;
        let rd = read_register::<F>(memory, a);
        let rs1 = read_register::<F>(memory, b);
        let bytes = (0..rs1)
            .map(|i| memory_read::<1>(memory, RV32_MEMORY_AS, mem_start + rd + i)[0])
            .collect::<Vec<u8>>();
        let peeked_str = String::from_utf8(bytes)?;
        print!("{peeked_str}");
        Ok(())
    }
}

/// HintRandom: Prepares random numbers for hinting.
pub struct HintRandomSubEx;

impl<F: PrimeField32> PhantomSubExecutor<F> for HintRandomSubEx {
    fn phantom_execute(
        &self,
        memory: &GuestMemory,
        streams: &mut Streams<F>,
        rng: &mut StdRng,
        _: PhantomDiscriminant,
        a: u32,
        _: u32,
        _: u16,
    ) -> eyre::Result<()> {
        let len = read_register::<F>(memory, a) as usize;
        streams.hint_stream.clear();
        streams
            .hint_stream
            .extend(std::iter::repeat_with(|| F::from_u8(rng.random::<u8>())).take(len * 4));
        Ok(())
    }
}

/// HintLoadByKey: Loads values from the KV store into input stream.
///
/// Note: `Streams::kv_store` was removed in openvm v2. This executor now
/// always returns an error because the feature is no longer supported.
pub struct HintLoadByKeySubEx;

impl<F: PrimeField32> PhantomSubExecutor<F> for HintLoadByKeySubEx {
    fn phantom_execute(
        &self,
        memory: &GuestMemory,
        _streams: &mut Streams<F>,
        _: &mut StdRng,
        _: PhantomDiscriminant,
        a: u32,
        b: u32,
        _: u16,
    ) -> eyre::Result<()> {
        let ptr = read_register::<F>(memory, a);
        let len = read_register::<F>(memory, b);
        let _key: Vec<u8> = (0..len)
            .map(|i| memory_read::<1>(memory, RV32_MEMORY_AS, ptr + i)[0])
            .collect();
        bail!("HintLoadByKey is no longer supported (kv_store removed in openvm v2)");
    }
}

/// ClockTimeGet: fills hint stream with 8 bytes of an incrementing nanosecond
/// timestamp. Starts at 1 second and increments by 1 ms per call.
/// Go's runtime expects monotonically increasing values from clock_time_get;
/// returning a constant causes infinite loops in time-based scheduling.
pub struct ClockTimeGetSubEx {
    counter: AtomicU64,
}

impl ClockTimeGetSubEx {
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(1_000_000_000),
        }
    }
}

impl<F: PrimeField32> PhantomSubExecutor<F> for ClockTimeGetSubEx {
    fn phantom_execute(
        &self,
        _: &GuestMemory,
        streams: &mut Streams<F>,
        _: &mut StdRng,
        _: PhantomDiscriminant,
        _: u32,
        _: u32,
        _: u16,
    ) -> eyre::Result<()> {
        let ns = self.counter.fetch_add(1_000_000, Ordering::Relaxed);
        streams.hint_stream.clear();
        streams
            .hint_stream
            .extend(ns.to_le_bytes().iter().map(|&b| F::from_u8(b)));
        Ok(())
    }
}

/// TraceSyscall: prints a WASI syscall name with sequence number to stderr.
pub struct TraceSyscallSubEx {
    counter: AtomicU32,
}

impl TraceSyscallSubEx {
    pub fn new() -> Self {
        Self {
            counter: AtomicU32::new(0),
        }
    }
}

impl<F: Field> PhantomSubExecutor<F> for TraceSyscallSubEx {
    fn phantom_execute(
        &self,
        _: &GuestMemory,
        _: &mut Streams<F>,
        _: &mut StdRng,
        _: PhantomDiscriminant,
        _: u32,
        _: u32,
        c_upper: u16,
    ) -> eyre::Result<()> {
        let seq = self.counter.fetch_add(1, Ordering::Relaxed);
        let name = match c_upper {
            0 => "args_sizes_get",
            1 => "args_get",
            2 => "environ_sizes_get",
            3 => "environ_get",
            4 => "fd_write",
            5 => "fd_read",
            6 => "fd_close",
            7 => "fd_fdstat_get",
            8 => "fd_fdstat_set_flags",
            9 => "fd_prestat_get",
            10 => "clock_time_get",
            11 => "random_get",
            12 => "proc_exit",
            13 => "poll_oneoff",
            14 => "fd_seek",
            15 => "fd_sync",
            16 => "sched_yield",
            17 => "path_open",
            _ => "unknown",
        };
        eprintln!("[wasi] #{seq} {name}");
        Ok(())
    }
}
