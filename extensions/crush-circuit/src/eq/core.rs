use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::*,
    system::memory::{MemoryAuxColsFactory, online::TracingMemory},
};
use openvm_circuit_primitives::utils::not;
use openvm_circuit_primitives_derive::{AlignedBorrow, AlignedBytesBorrow};
use openvm_crush_transpiler::EqOpcode;
use openvm_instructions::{LocalOpcode, instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    BaseAirWithPublicValues, ColumnsAir,
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};
use struct_reflection::{StructReflection, StructReflectionHelper};
use strum::IntoEnumIterator;

// ======================== Core Columns ========================

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct EqCoreCols<T, const NUM_LIMBS: usize> {
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    /// Boolean result: 1 if condition holds, 0 otherwise.
    pub cmp_result: T,

    pub opcode_eq_flag: T,
    pub opcode_neq_flag: T,

    /// For proving inequality: at the first position i where b[i] != c[i],
    /// diff_inv_marker[i] = inverse(b[i] - c[i]). All other positions are 0.
    /// When b == c, all positions are 0.
    pub diff_inv_marker: [T; NUM_LIMBS],
}

// ======================== Core AIR ========================

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct EqCoreAir<const NUM_LIMBS: usize> {
    offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize> BaseAir<F> for EqCoreAir<NUM_LIMBS> {
    fn width(&self) -> usize {
        EqCoreCols::<F, NUM_LIMBS>::width()
    }
}

impl<F: Field, const NUM_LIMBS: usize> ColumnsAir<F> for EqCoreAir<NUM_LIMBS> {
    fn columns(&self) -> Option<Vec<String>> {
        EqCoreCols::<F, NUM_LIMBS>::struct_reflection()
    }
}

impl<F: Field, const NUM_LIMBS: usize> BaseAirWithPublicValues<F> for EqCoreAir<NUM_LIMBS> {}

impl<AB, I, const NUM_LIMBS: usize> VmCoreAir<AB, I> for EqCoreAir<NUM_LIMBS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &EqCoreCols<_, NUM_LIMBS> = local.borrow();
        let flags = [cols.opcode_eq_flag, cols.opcode_neq_flag];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.cmp_result);

        let b = &cols.b;
        let c = &cols.c;
        let inv_marker = &cols.diff_inv_marker;

        // 1 if claiming values are equal, 0 otherwise.
        // For EQ: cmp_result=1 means equal → cmp_eq=1.
        // For NEQ: cmp_result=0 means equal → cmp_eq=1.
        let cmp_eq =
            cols.cmp_result * cols.opcode_eq_flag + not(cols.cmp_result) * cols.opcode_neq_flag;
        let mut sum = cmp_eq.clone();

        // Equality proof via multiplicative inverses:
        // - If b == c: cmp_eq must be 1, all (b[i] - c[i]) are 0, so sum = 1. ✓
        //   The second constraint cmp_eq * (b[i] - c[i]) = 0 holds since differences are 0.
        // - If b != c: cmp_eq must be 0, and at some position i where b[i] != c[i],
        //   inv_marker[i] = inverse(b[i] - c[i]), so (b[i] - c[i]) * inv_marker[i] = 1,
        //   making sum = 0 + 1 = 1. ✓
        //   The second constraint trivially holds since cmp_eq = 0.
        for i in 0..NUM_LIMBS {
            sum += (b[i] - c[i]) * inv_marker[i];
            builder.assert_zero(cmp_eq.clone() * (b[i] - c[i]));
        }
        builder.when(is_valid.clone()).assert_one(sum);

        let expected_opcode = flags
            .iter()
            .zip(EqOpcode::iter())
            .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_u8(opcode as u8)
            })
            + AB::Expr::from_usize(self.offset);

        // Output: cmp_result in the first limb, zeros elsewhere.
        let mut a: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        a[0] = cols.cmp_result.into();

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [a].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: expected_opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

// ======================== Core Record ========================

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct EqCoreRecord<const NUM_LIMBS: usize> {
    pub b: [u8; NUM_LIMBS],
    pub c: [u8; NUM_LIMBS],
    pub local_opcode: u8,
}

// ======================== Preflight Executor ========================

#[derive(Clone, Copy, derive_new::new)]
pub struct EqExecutorInner<A, const NUM_LIMBS: usize> {
    adapter: A,
    pub offset: usize,
}

impl<F, A, RA, const NUM_LIMBS: usize> PreflightExecutor<F, RA> for EqExecutorInner<A, NUM_LIMBS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
            WriteData: From<[[u8; NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
            'buf,
            EmptyAdapterCoreLayout<F, A>,
            (A::RecordMut<'buf>, &'buf mut EqCoreRecord<NUM_LIMBS>),
        >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", EqOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        core_record.b = rs1;
        core_record.c = rs2;
        core_record.local_opcode = opcode.local_opcode_idx(self.offset) as u8;

        let is_eq = core_record.local_opcode == EqOpcode::EQ as u8;
        let cmp_result = if is_eq { rs1 == rs2 } else { rs1 != rs2 };

        let mut output = [0u8; NUM_LIMBS];
        output[0] = cmp_result as u8;

        self.adapter.write(
            state.memory,
            instruction,
            [output].into(),
            &mut adapter_record,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

// ======================== Trace Filler ========================

#[derive(Clone, derive_new::new)]
pub struct EqFiller<A, const NUM_LIMBS: usize> {
    adapter: A,
    pub offset: usize,
}

impl<F, A, const NUM_LIMBS: usize> TraceFiller<F> for EqFiller<A, NUM_LIMBS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // EqCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid EqCoreRecord written by the executor
        // during trace generation
        let record: &EqCoreRecord<NUM_LIMBS> = unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut EqCoreCols<F, NUM_LIMBS> = core_row.borrow_mut();

        let is_eq = record.local_opcode == EqOpcode::EQ as u8;
        let (cmp_result, diff_idx, diff_inv_val) =
            run_eq::<F, NUM_LIMBS>(is_eq, &record.b, &record.c);

        core_row.diff_inv_marker = [F::ZERO; NUM_LIMBS];
        core_row.diff_inv_marker[diff_idx] = diff_inv_val;

        core_row.opcode_eq_flag = F::from_bool(is_eq);
        core_row.opcode_neq_flag = F::from_bool(!is_eq);
        core_row.cmp_result = F::from_bool(cmp_result);
        // Write c before b: the record overlaps core_row memory, and writing
        // core_row.b (bytes 0..15) would overwrite record.c (bytes 4..7).
        core_row.c = record.c.map(F::from_u8);
        core_row.b = record.b.map(F::from_u8);
    }
}

// ======================== Helper ========================

/// Returns (cmp_result, diff_idx, diff_inv_val).
///
/// When values are equal: diff_idx = 0, diff_inv_val = 0.
/// When values differ at position i: diff_idx = i, diff_inv_val = inverse(b[i] - c[i]).
#[inline(always)]
fn run_eq<F, const NUM_LIMBS: usize>(
    is_eq: bool,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> (bool, usize, F)
where
    F: PrimeField32,
{
    for i in 0..NUM_LIMBS {
        if x[i] != y[i] {
            return (!is_eq, i, (F::from_u8(x[i]) - F::from_u8(y[i])).inverse());
        }
    }
    (is_eq, 0, F::ZERO)
}
