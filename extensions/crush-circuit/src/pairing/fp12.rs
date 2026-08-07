use std::{array::from_fn, cell::RefCell, rc::Rc};

use openvm_algebra_circuit::Fp2;
use openvm_mod_circuit_builder::{ExprBuilder, FieldVariable};

/// Field extension Fp12 defined with coefficients in Fp2.
/// Represents the element `c0 + c1 w + ... + c5 w^5` in Fp12.
/// Fp6-equivalent coefficients are c0: (c0, c2, c4), c1: (c1, c3, c5).
pub struct Fp12 {
    pub c: [Fp2; 6],
}

impl Fp12 {
    pub fn new(builder: Rc<RefCell<ExprBuilder>>) -> Self {
        let c = from_fn(|_| Fp2::new(builder.clone()));
        Fp12 { c }
    }

    pub fn save(&mut self) -> [usize; 12] {
        self.c
            .each_mut()
            .map(|c| c.save())
            .concat()
            .try_into()
            .unwrap()
    }

    pub fn save_output(&mut self) {
        for c in self.c.iter_mut() {
            c.save_output();
        }
    }

    pub fn add(&mut self, other: &mut Fp12) -> Fp12 {
        Fp12 {
            c: from_fn(|i| self.c[i].add(&mut other.c[i])),
        }
    }

    pub fn sub(&mut self, other: &mut Fp12) -> Fp12 {
        Fp12 {
            c: from_fn(|i| self.c[i].sub(&mut other.c[i])),
        }
    }

    pub fn mul(&mut self, other: &mut Fp12, xi: [isize; 2]) -> Fp12 {
        let c = from_fn(|i| {
            let mut sum = self.c[0].mul(&mut other.c[i]);
            for j in 1..=5.min(i) {
                let k = i - j;
                sum = sum.add(&mut self.c[j].mul(&mut other.c[k]));
            }
            let mut sum_hi: Option<Fp2> = None;
            for j in (i + 1)..=5 {
                let k = 6 + i - j;
                let mut term = self.c[j].mul(&mut other.c[k]);
                if let Some(mut running_sum) = sum_hi {
                    sum_hi = Some(running_sum.add(&mut term));
                } else {
                    sum_hi = Some(term);
                }
            }
            if let Some(mut sum_hi) = sum_hi {
                sum = sum.add(&mut sum_hi.int_mul(xi));
            }
            sum.save();
            sum
        });
        Fp12 { c }
    }

    /// Multiply self by `x0 + x1 w + x2 w^2 + x3 w^3 + x4 w^4` in Fp12.
    pub fn mul_by_01234(
        &mut self,
        x0: &mut Fp2,
        x1: &mut Fp2,
        x2: &mut Fp2,
        x3: &mut Fp2,
        x4: &mut Fp2,
        xi: [isize; 2],
    ) -> Fp12 {
        let c0 = self.c[0].mul(x0).add(
            &mut self.c[2]
                .mul(x4)
                .add(&mut self.c[3].mul(x3))
                .add(&mut self.c[4].mul(x2))
                .add(&mut self.c[5].mul(x1))
                .int_mul(xi),
        );

        let c1 = self.c[0].mul(x1).add(&mut self.c[1].mul(x0)).add(
            &mut self.c[3]
                .mul(x4)
                .add(&mut self.c[4].mul(x3))
                .add(&mut self.c[5].mul(x2))
                .int_mul(xi),
        );

        let c2 = self.c[0]
            .mul(x2)
            .add(&mut self.c[1].mul(x1))
            .add(&mut self.c[2].mul(x0))
            .add(&mut self.c[4].mul(x4).add(&mut self.c[5].mul(x3)).int_mul(xi));

        let c3 = self.c[0]
            .mul(x3)
            .add(&mut self.c[1].mul(x2))
            .add(&mut self.c[2].mul(x1))
            .add(&mut self.c[3].mul(x0))
            .add(&mut self.c[5].mul(x4).int_mul(xi));

        let c4 = self.c[0]
            .mul(x4)
            .add(&mut self.c[1].mul(x3))
            .add(&mut self.c[2].mul(x2))
            .add(&mut self.c[3].mul(x1))
            .add(&mut self.c[4].mul(x0));

        let c5 = self.c[1]
            .mul(x4)
            .add(&mut self.c[2].mul(x3))
            .add(&mut self.c[3].mul(x2))
            .add(&mut self.c[4].mul(x1))
            .add(&mut self.c[5].mul(x0));

        Fp12 {
            c: [c0, c1, c2, c3, c4, c5],
        }
    }

    /// Multiply `self` by `x0 + x2 w^2 + x3 w^3 + x4 w^4 + x5 w^5` in Fp12.
    pub fn mul_by_02345(
        &mut self,
        x0: &mut Fp2,
        x2: &mut Fp2,
        x3: &mut Fp2,
        x4: &mut Fp2,
        x5: &mut Fp2,
        xi: [isize; 2],
    ) -> Fp12 {
        let c0 = self.c[0].mul(x0).add(
            &mut self.c[1]
                .mul(x5)
                .add(&mut self.c[2].mul(x4))
                .add(&mut self.c[3].mul(x3))
                .add(&mut self.c[4].mul(x2))
                .int_mul(xi),
        );

        let c1 = self.c[1].mul(x0).add(
            &mut self.c[2]
                .mul(x5)
                .add(&mut self.c[3].mul(x4))
                .add(&mut self.c[4].mul(x3))
                .add(&mut self.c[5].mul(x2))
                .int_mul(xi),
        );

        let c2 = self.c[0].mul(x2).add(&mut self.c[2].mul(x0)).add(
            &mut self.c[3]
                .mul(x5)
                .add(&mut self.c[4].mul(x4))
                .add(&mut self.c[5].mul(x3))
                .int_mul(xi),
        );

        let c3 = self.c[0]
            .mul(x3)
            .add(&mut self.c[1].mul(x2))
            .add(&mut self.c[3].mul(x0))
            .add(&mut self.c[4].mul(x5).add(&mut self.c[5].mul(x4)).int_mul(xi));

        let c4 = self.c[0]
            .mul(x4)
            .add(&mut self.c[1].mul(x3))
            .add(&mut self.c[2].mul(x2))
            .add(&mut self.c[4].mul(x0))
            .add(&mut self.c[5].mul(x5).int_mul(xi));

        let c5 = self.c[0]
            .mul(x5)
            .add(&mut self.c[1].mul(x4))
            .add(&mut self.c[2].mul(x3))
            .add(&mut self.c[3].mul(x2))
            .add(&mut self.c[5].mul(x0));

        Fp12 {
            c: [c0, c1, c2, c3, c4, c5],
        }
    }

    pub fn div(&mut self, _other: &mut Fp12, _xi: [isize; 2]) -> Fp12 {
        unimplemented!()
    }

    pub fn scalar_mul(&mut self, fp: &mut FieldVariable) -> Fp12 {
        Fp12 {
            c: from_fn(|i| self.c[i].scalar_mul(fp)),
        }
    }
}
