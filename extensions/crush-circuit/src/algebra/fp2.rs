use std::{cell::RefCell, rc::Rc};

use openvm_mod_circuit_builder::{ExprBuilder, FieldVariable, SymbolicExpr};

/// Quadratic field extension of `Fp` defined by `Fp2 = Fp[u]/(1 + u^2)`. Assumes that `-1` is not a
/// quadratic residue in `Fp`, which is equivalent to `p` being congruent to `3 (mod 4)`.
/// Extends Mod Builder to work with Fp2 variables.
#[derive(Clone)]
pub struct Fp2 {
    pub c0: FieldVariable,
    pub c1: FieldVariable,
}

impl Fp2 {
    pub fn new(builder: Rc<RefCell<ExprBuilder>>) -> Self {
        let c0 = ExprBuilder::new_input(builder.clone());
        let c1 = ExprBuilder::new_input(builder.clone());
        Fp2 { c0, c1 }
    }

    pub fn new_var(builder: Rc<RefCell<ExprBuilder>>) -> ((usize, usize), Fp2) {
        let (c0_idx, c0) = builder.borrow_mut().new_var();
        let (c1_idx, c1) = builder.borrow_mut().new_var();
        let fp2 = Fp2 {
            c0: FieldVariable::from_var(builder.clone(), c0),
            c1: FieldVariable::from_var(builder.clone(), c1),
        };
        ((c0_idx, c1_idx), fp2)
    }

    pub fn save(&mut self) -> [usize; 2] {
        let c0_idx = self.c0.save();
        let c1_idx = self.c1.save();
        [c0_idx, c1_idx]
    }

    pub fn save_output(&mut self) {
        self.c0.save_output();
        self.c1.save_output();
    }

    pub fn add(&mut self, other: &mut Fp2) -> Fp2 {
        Fp2 {
            c0: &mut self.c0 + &mut other.c0,
            c1: &mut self.c1 + &mut other.c1,
        }
    }

    pub fn sub(&mut self, other: &mut Fp2) -> Fp2 {
        Fp2 {
            c0: &mut self.c0 - &mut other.c0,
            c1: &mut self.c1 - &mut other.c1,
        }
    }

    pub fn mul(&mut self, other: &mut Fp2) -> Fp2 {
        let c0 = &mut self.c0 * &mut other.c0 - &mut self.c1 * &mut other.c1;
        let c1 = &mut self.c0 * &mut other.c1 + &mut self.c1 * &mut other.c0;
        Fp2 { c0, c1 }
    }

    pub fn square(&mut self) -> Fp2 {
        let c0 = self.c0.square() - self.c1.square();
        let c1 = (&mut self.c0 * &mut self.c1).int_mul(2);
        Fp2 { c0, c1 }
    }

    pub fn div(&mut self, other: &mut Fp2) -> Fp2 {
        let builder = self.c0.builder.borrow();
        let prime = builder.prime.clone();
        let limb_bits = builder.limb_bits;
        let num_limbs = builder.num_limbs;
        let proper_max = builder.proper_max().clone();
        drop(builder);

        // These are dummy variables, will be replaced later so the index within it doesn't matter.
        // We use these to check if we need to save self/other first.
        let fake_z0 = SymbolicExpr::Var(0);
        let fake_z1 = SymbolicExpr::Var(1);

        // Compute should not be affected by whether auto save is triggered.
        // So we must do compute first.
        // Compute z0
        let compute_denom = &other.c0.expr * &other.c0.expr + &other.c1.expr * &other.c1.expr;
        let compute_z0_nom = &self.c0.expr * &other.c0.expr + &self.c1.expr * &other.c1.expr;
        let compute_z0 = &compute_z0_nom / &compute_denom;
        // Compute z1
        let compute_z1_nom = &self.c1.expr * &other.c0.expr - &self.c0.expr * &other.c1.expr;
        let compute_z1 = &compute_z1_nom / &compute_denom;

        // We will constrain
        //  (1) x0 = y0*z0 - y1*z1 and
        //  (2) x1 = y1*z0 + y0*z1
        // which implies z0 and z1 are computed as above.
        // Observe (1)*y0 + (2)*y1 yields x0*y0 + x1*y1 = z0(y0^2 + y1^2) and so z0 = (x0*y0 +
        // x1*y1) / (y0^2 + y1^2) as needed. Observe (1)*(-y1) + (2)*y0 yields x1*y0 - x0*y1
        // = z1(y0^2 + y1^2) and so z1 = (x1*y0 - x0*y1) / (y0^2 + y1^2) as needed.

        // Constraint 1: x0 = y0*z0 - y1*z1
        let constraint1 = &self.c0.expr - &other.c0.expr * &fake_z0 + &other.c1.expr * &fake_z1;
        let carry_bits =
            constraint1.constraint_carry_bits_with_pq(&prime, limb_bits, num_limbs, &proper_max);
        if carry_bits > self.c0.max_carry_bits {
            self.save();
        }
        let constraint1 = &self.c0.expr - &other.c0.expr * &fake_z0 + &other.c1.expr * &fake_z1;
        let carry_bits =
            constraint1.constraint_carry_bits_with_pq(&prime, limb_bits, num_limbs, &proper_max);
        if carry_bits > self.c0.max_carry_bits {
            other.save();
        }

        // Constraint 2: x1 = y1*z0 + y0*z1
        let constraint2 = &self.c1.expr - &other.c1.expr * &fake_z0 - &other.c0.expr * &fake_z1;
        let carry_bits =
            constraint2.constraint_carry_bits_with_pq(&prime, limb_bits, num_limbs, &proper_max);
        if carry_bits > self.c0.max_carry_bits {
            self.save();
        }
        let constraint2 = &self.c1.expr - &other.c1.expr * &fake_z0 - &other.c0.expr * &fake_z1;
        let carry_bits =
            constraint2.constraint_carry_bits_with_pq(&prime, limb_bits, num_limbs, &proper_max);
        if carry_bits > self.c0.max_carry_bits {
            other.save();
        }

        let mut builder = self.c0.builder.borrow_mut();
        let (z0_idx, z0) = builder.new_var();
        let (z1_idx, z1) = builder.new_var();
        let constraint1 = &self.c0.expr - &other.c0.expr * &z0 + &other.c1.expr * &z1;
        let constraint2 = &self.c1.expr - &other.c1.expr * &z0 - &other.c0.expr * &z1;
        builder.set_compute(z0_idx, compute_z0);
        builder.set_compute(z1_idx, compute_z1);
        builder.set_constraint(z0_idx, constraint1);
        builder.set_constraint(z1_idx, constraint2);
        drop(builder);

        let z0_var = FieldVariable::from_var(self.c0.builder.clone(), z0);
        let z1_var = FieldVariable::from_var(self.c0.builder.clone(), z1);
        Fp2 {
            c0: z0_var,
            c1: z1_var,
        }
    }

    pub fn scalar_mul(&mut self, fp: &mut FieldVariable) -> Fp2 {
        Fp2 {
            c0: &mut self.c0 * fp,
            c1: &mut self.c1 * fp,
        }
    }

    pub fn int_add(&mut self, c: [isize; 2]) -> Fp2 {
        Fp2 {
            c0: self.c0.int_add(c[0]),
            c1: self.c1.int_add(c[1]),
        }
    }

    // c is like a Fp2, but with both c0 and c1 being very small numbers.
    pub fn int_mul(&mut self, c: [isize; 2]) -> Fp2 {
        Fp2 {
            c0: self.c0.int_mul(c[0]) - self.c1.int_mul(c[1]),
            c1: self.c0.int_mul(c[1]) + self.c1.int_mul(c[0]),
        }
    }

    pub fn neg(&mut self) -> Fp2 {
        self.int_mul([-1, 0])
    }

    pub fn select(flag_id: usize, a: &Fp2, b: &Fp2) -> Fp2 {
        Fp2 {
            c0: FieldVariable::select(flag_id, &a.c0, &b.c0),
            c1: FieldVariable::select(flag_id, &a.c1, &b.c1),
        }
    }
}
