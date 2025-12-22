//! Known bits analysis.
//!
//! Data-flow analysis to gain information about bits.
//! We use similar ideas to PyPy:
//! https://pypy.org/posts/2024/08/toy-knownbits.html
//!

use crate::compile::{
    j2::{
        hir::*,
        opt::opt::{Opt, OptOutcome},
    },
    jitc_yk::arbbitint::ArbBitInt,
};

/// Known bits for a single value.
///
/// This uses the same bit representation found in
/// https://pypy.org/posts/2024/08/toy-knownbits.html#the-knownbits-abstract-domain
///
/// In short:
/// one     unknown     knownbit
/// 0       0           0
/// 1       0           1
/// 0       1           ?
/// 1       1           contradiction
///
/// To ensure monotonicity, state always goes upwards in this lattice.
///           contradiction
///             /      \
///             1      0
///             \      /
///                ?
#[derive(Clone)]
pub struct KnownBitValue {
    ones: ArbBitInt,
    pub unknowns: ArbBitInt,
}

/// Full credits to the PyPy blog post for some of the functions here:
/// https://pypy.org/posts/2024/08/toy-knownbits.html#the-knownbits-abstract-domain
impl KnownBitValue {
    fn from_constant(num: &ArbBitInt) -> Self {
        let bitw = num.bitw();
        KnownBitValue {
            ones: ArbBitInt::from_u64(bitw, num.to_zero_ext_u64().unwrap()),
            unknowns: ArbBitInt::from_u64(bitw, 0),
        }
    }

    pub fn unknown(bitw: u32) -> Self {
        KnownBitValue {
            ones: ArbBitInt::from_u64(bitw, 0),
            unknowns: ArbBitInt::from_u64(bitw, !0u64),
        }
    }

    fn as_int(&self) -> ArbBitInt {
        assert!(self.all_known());
        self.ones.clone()
    }

    fn all_known(&self) -> bool {
        self.unknowns.count_ones() == 0
    }

    /// This usually means either a bug in our code or malformed HIR.
    /// In which case, it's time to bail.
    pub fn contradiction(&self) -> bool {
        self.ones.bitand(&self.unknowns).count_ones() != 0
    }

    /// Return an integer where the known bits are set
    fn knowns(&self) -> ArbBitInt {
        self.unknowns.bitneg()
    }

    /// Returns an integer where the places that are known zeros have a bit set
    fn zeroes(&self) -> ArbBitInt {
        self.knowns().bitand(&self.ones.bitneg())
    }

    fn known_ones(&self) -> ArbBitInt {
        self.knowns().bitand(&self.ones)
    }

    fn and(&self, other: &KnownBitValue) -> KnownBitValue {
        let set_ones = self.ones.bitand(&other.ones);
        let set_zeroes = self.zeroes().bitor(&other.zeroes());
        let unknowns = self
            .unknowns
            .bitor(&other.unknowns)
            .bitand(&set_zeroes.bitneg());
        KnownBitValue {
            ones: set_ones,
            unknowns,
        }
    }

    fn or(&self, other: &KnownBitValue) -> KnownBitValue {
        let set_ones = self.ones.bitor(&other.ones);
        let unknowns = self
            .unknowns
            .bitor(&other.unknowns)
            .bitand(&set_ones.bitneg());
        KnownBitValue {
            ones: set_ones,
            unknowns,
        }
    }
}

/// For the known bits analysis, we map each SSA value to a KnownBitValue.
pub(super) struct KnownBits {
    pub(super) ssa_bits: Vec<KnownBitValue>,
    pub(super) contradiction: bool,
}

impl KnownBits {
    /// Create an empty known bits analysis object.
    pub(super) fn new() -> Self {
        KnownBits {
            ssa_bits: Vec::new(),
            contradiction: false,
        }
    }

    pub(super) fn known_bits_step(opt: &mut Opt, inst: Inst) -> OptOutcome {
        if opt.known_bits_contradiction() {
            return OptOutcome::Rewritten(inst);
        }
        let inst_copy = inst.clone();
        let outcome = match inst {
            Inst::And(x) => opt_and(opt, x),
            Inst::Const(x) => opt_const(opt, x),
            Inst::Or(x) => opt_or(opt, x),
            _ => OptOutcome::Rewritten(inst),
        };
        if opt.known_bits_contradiction() {
            OptOutcome::Rewritten(inst_copy)
        } else {
            outcome
        }
    }

    pub(super) fn get_inst(&mut self, iidx: InstIdx) -> KnownBitValue {
        if iidx > self.ssa_bits.len() {
            panic!("Corrupted SSA form---use not dominated by def.")
        }
        self.ssa_bits[iidx.index()].clone()
    }

    pub(super) fn set_inst_known_bit(&mut self, value: KnownBitValue) {
        let len = self.ssa_bits.len() - 1;
        self.ssa_bits[len] = value;
    }
}

fn opt_and(opt: &mut Opt, mut inst: And) -> OptOutcome {
    inst.canonicalise(opt);
    let And { tyidx, lhs, rhs } = inst;
    let lhs_b = opt.as_known_bits(lhs);
    let rhs_b = opt.as_known_bits(rhs);
    let res = lhs_b.and(&rhs_b);
    opt.set_known_bits(res.clone());

    // If we know the output's bits, emit that.
    if res.all_known() {
        return OptOutcome::Rewritten(Inst::Const(Const {
            tyidx,
            kind: ConstKind::Int(res.as_int()),
        }));
    }

    // lhs = any and rhs = constant
    // If no new information is gained, that means this
    // op was useless.
    if rhs_b.all_known() &&
        (rhs_b.zeroes().bitand(&lhs_b.known_ones().bitor(&lhs_b.unknowns)).count_ones() == 0) {
        return OptOutcome::Equiv(lhs);
    }

    OptOutcome::Rewritten(inst.into())
}

fn opt_const(opt: &mut Opt, inst: Const) -> OptOutcome {
    let Const { tyidx: _, kind } = &inst;
    if let ConstKind::Int(kind) = kind {
        opt.set_known_bits(KnownBitValue::from_constant(kind))
    }
    OptOutcome::Rewritten(inst.into())
}

fn opt_or(opt: &mut Opt, mut inst: Or) -> OptOutcome {
    inst.canonicalise(opt);
    let Or {
        tyidx,
        lhs,
        rhs,
        disjoint: _,
    } = inst;
    let lhs_b = opt.as_known_bits(lhs);
    let rhs_b = opt.as_known_bits(rhs);
    let res = lhs_b.or(&rhs_b);
    opt.set_known_bits(res.clone());

    // If we know the output's bits, emit that.
    if res.all_known() {
        return OptOutcome::Rewritten(Inst::Const(Const {
            tyidx,
            kind: ConstKind::Int(res.as_int()),
        }));
    }

    // lhs = any and rhs = constant
    // If no new information is gained, that means this
    // op was useless.
    if rhs_b.all_known() &&
        (rhs_b.known_ones().bitand(&lhs_b.zeroes().bitor(&lhs_b.unknowns)).count_ones() == 0) {
        return OptOutcome::Equiv(lhs);
    }

    OptOutcome::Rewritten(inst.into())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::compile::j2::opt::strength_fold::strength_fold;
    use crate::compile::j2::opt::{known_bits::KnownBits, opt::test::opt_and_test};

    fn test_known_bits(mod_s: &str, ptn: &str) {
        opt_and_test(
            mod_s,
            |opt, mut inst| {
                opt.known_bits_new_inst(&inst);
                inst.canonicalise(opt);
                match KnownBits::known_bits_step(opt, inst) {
                    OptOutcome::Rewritten(inst) => strength_fold(opt, inst),
                    outcome => outcome,
                }
            },
            ptn,
        );
    }

    #[test]
    fn opt_and() {
        // any = any & 01
        // any = any & 11 <-- should be removed
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i8 = 3
          %3: i8 = and %0, %1
          %4: i8 = and %3, %2
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = 1
          %2: i8 = 3
          %3: i8 = and %0, %1
          blackbox %3
        ",
        );

        // any = any & 11
        // any = any & 01 <-- should not be removed
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 3
          %2: i8 = 1
          %3: i8 = and %0, %1
          %4: i8 = and %3, %2
          blackbox %4
        ",
            "
          %0: i8 = arg
          %1: i8 = 3
          %2: i8 = 1
          %3: i8 = and %0, %1
          %4: i8 = and %3, %2
          blackbox %4
        ",
        );

        // any & any
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = arg [reg]
          %2: i8 = and %0, %1
          blackbox %2
        ",
            "
          %0: i8 = arg
          %1: i8 = arg
          %2: i8 = and %0, %1
          blackbox %2
        ",
        );
    }

    #[test]
    fn opt_or() {
        // Test bitwise or
        // any = any | 01
        // any = any | 10
        // any = any | 11 <-- should be removed
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i8 = 2
          %3: i8 = 3
          %4: i8 = or %0, %1
          %5: i8 = or %4, %2
          %6: i8 = or %5, %3
          blackbox %6
        ",
            "
          %0: i8 = arg
          %1: i8 = 1
          %2: i8 = 2
          %3: i8 = 3
          %4: i8 = or %0, %1
          %5: i8 = or %4, %2
          blackbox %5
        ",
        );

        // Test bitwise or
        // any = any | 01
        // any = any | 10  <-- should not be removed
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i8 = 2
          %3: i8 = 3
          %4: i8 = or %0, %1
          %5: i8 = or %4, %2
          blackbox %5
        ",
            "
          %0: i8 = arg
          %1: i8 = 1
          %2: i8 = 2
          %3: i8 = 3
          %4: i8 = or %0, %1
          %5: i8 = or %4, %2
          blackbox %5
        ",
        );
    }

    #[test]
    fn opt_and_or() {
        // Test that and is eliminated
        // if a bit is set by or.
        test_known_bits(
            "
          %0: i8 = arg [reg]
          %1: i8 = 1
          %2: i8 = or %0, %1
          %3: i8 = and %2, %1
          blackbox %3
        ",
            "
          %0: i8 = arg
          %1: i8 = 1
          %2: i8 = or %0, %1
          %3: i8 = 1
          blackbox %3
        ",
        );
    }

    #[test]
    fn with_intermediate() {
        // Test that other instructions stay around
        test_known_bits(
            "
          %0: ptr = arg [reg]
          %1: ptr = ptradd %0, 8
          %2: i8 = load %1
          %3: i8 = 15
          %4: i8 = and %2, %3
          %5: i8 = 0
          %6: i1 = icmp eq %4, %5
          blackbox %6
        ",
            "
          %0: ptr = arg
          %1: ptr = ptradd %0, 8
          %2: i8 = load %1
          %3: i8 = 15
          %4: i8 = and %2, %3
          %5: i8 = 0
          %6: i1 = icmp eq %4, %5
          blackbox %6
        ",
        );

        test_known_bits(
            "
          %0: ptr = arg [reg]
          %1: i8 = load %0
          %2: i8 = 15
          %3: i8 = and %1, %2
          %4: i8 = 0
          %5: i1 = icmp eq %3, %4
          blackbox %5
        ",
            "
          %0: ptr = arg
          %1: i8 = load %0
          %2: i8 = 15
          %3: i8 = and %1, %2
          %4: i8 = 0
          %5: i1 = icmp eq %3, %4
          blackbox %5
        ",
        );

        test_known_bits(
            "
          %0: ptr = arg [reg]
          %1: i32 = arg [reg]
          %2: ptr = ptradd %0, 4
          %3: i32 = 127
          %4: i32 = and %1, %3
          %5: i32 = 34
          %6: i1 = icmp eq %4, %5
          blackbox %6
        ",
            "
          %0: ptr = arg
          %1: i32 = arg
          %2: ptr = ptradd %0, 4
          %3: i32 = 127
          %4: i32 = and %1, %3
          %5: i32 = 34
          %6: i1 = icmp eq %4, %5
          blackbox %6
        ",
        );
    }
}
