import jax.numpy as jnp
from dl2.diffsat import (
    diffsat_theta,
    diffsat_delta,
    BConstant,
    GT,
    LT,
    EQ,
    GEQ,
    LEQ,
    AND,
    OR,
    IFTHEN,
    NEGATE,
)


# Helper function to create scalar arrays
def scalar(value):
    return jnp.array(value)


# Test diffsat_theta
def test_diffsat_theta_scalars():
    assert diffsat_theta(scalar(5.0), scalar(3.0)) == scalar(2.0)
    assert diffsat_theta(scalar(3.0), scalar(5.0)) == scalar(2.0)
    assert diffsat_theta(scalar(3.0), scalar(3.0)) == scalar(0.0)


# Test diffsat_delta
def test_diffsat_delta_scalars():
    assert diffsat_delta(scalar(5.0), scalar(3.0)) == scalar(2.0)
    assert diffsat_delta(scalar(3.0), scalar(5.0)) == scalar(-2.0)
    assert diffsat_delta(scalar(3.0), scalar(3.0)) == scalar(0.0)


# Test BConstant
def test_bconstant_scalars():
    assert BConstant(scalar(1.0)).loss() == scalar(0.0)
    assert BConstant(scalar(1.0)).satisfy() == scalar(1.0)
    assert BConstant(scalar(0.0)).loss() == scalar(1.0)
    assert BConstant(scalar(0.0)).satisfy() == scalar(0.0)


# Test GT
def test_gt_scalars():
    assert GT(scalar(5.0), scalar(3.0)).loss() == scalar(0.0)
    assert GT(scalar(5.0), scalar(3.0)).satisfy() == scalar(1.0)
    assert GT(scalar(3.0), scalar(5.0)).loss() > scalar(0.0)
    assert GT(scalar(3.0), scalar(5.0)).satisfy() == scalar(0.0)
    assert GT(scalar(3.0), scalar(3.0)).loss() > scalar(0.0)
    assert GT(scalar(3.0), scalar(3.0)).satisfy() == scalar(0.0)


# Test LT
def test_lt_scalars():
    assert LT(scalar(3.0), scalar(5.0)).loss() == scalar(0.0)
    assert LT(scalar(3.0), scalar(5.0)).satisfy() == scalar(1.0)
    assert LT(scalar(5.0), scalar(3.0)).loss() > scalar(0.0)
    assert LT(scalar(5.0), scalar(3.0)).satisfy() == scalar(0.0)
    assert LT(scalar(3.0), scalar(3.0)).loss() > scalar(0.0)
    assert LT(scalar(3.0), scalar(3.0)).satisfy() == scalar(0.0)


# Test EQ
def test_eq_scalars():
    assert EQ(scalar(3.0), scalar(3.0)).loss() == scalar(0.0)
    assert EQ(scalar(3.0), scalar(3.0)).satisfy() == scalar(1.0)
    assert EQ(scalar(5.0), scalar(3.0)).loss() > scalar(0.0)
    assert EQ(scalar(5.0), scalar(3.0)).satisfy() == scalar(0.0)


# Test GEQ
def test_geq_scalars():
    assert GEQ(scalar(5.0), scalar(3.0)).loss() == scalar(0.0)
    assert GEQ(scalar(5.0), scalar(3.0)).satisfy() == scalar(1.0)
    assert GEQ(scalar(3.0), scalar(3.0)).loss() == scalar(0.0)
    assert GEQ(scalar(3.0), scalar(3.0)).satisfy() == scalar(1.0)
    assert GEQ(scalar(3.0), scalar(5.0)).loss() > scalar(0.0)
    assert GEQ(scalar(3.0), scalar(5.0)).satisfy() == scalar(0.0)


# Test LEQ
def test_leq_scalars():
    assert LEQ(scalar(3.0), scalar(5.0)).loss() == scalar(0.0)
    assert LEQ(scalar(3.0), scalar(5.0)).satisfy() == scalar(1.0)
    assert LEQ(scalar(3.0), scalar(3.0)).loss() == scalar(0.0)
    assert LEQ(scalar(3.0), scalar(3.0)).satisfy() == scalar(1.0)
    assert LEQ(scalar(5.0), scalar(3.0)).loss() > scalar(0.0)
    assert LEQ(scalar(5.0), scalar(3.0)).satisfy() == scalar(0.0)


# Test AND
def test_and_scalars():
    assert AND(
        [GT(scalar(5.0), scalar(3.0)), LT(scalar(3.0), scalar(5.0))]
    ).loss() == scalar(0.0)
    assert AND(
        [GT(scalar(5.0), scalar(3.0)), LT(scalar(3.0), scalar(5.0))]
    ).satisfy() == scalar(1.0)
    assert AND(
        [GT(scalar(5.0), scalar(3.0)), LT(scalar(5.0), scalar(3.0))]
    ).loss() > scalar(0.0)
    assert AND(
        [GT(scalar(5.0), scalar(3.0)), LT(scalar(5.0), scalar(3.0))]
    ).satisfy() == scalar(0.0)


# Test OR
def test_or_scalars():
    assert OR(
        [GT(scalar(5.0), scalar(3.0)), LT(scalar(3.0), scalar(5.0))]
    ).loss() == scalar(0.0)
    assert OR(
        [GT(scalar(5.0), scalar(3.0)), LT(scalar(3.0), scalar(5.0))]
    ).satisfy() == scalar(1.0)
    assert OR(
        [GT(scalar(3.0), scalar(5.0)), LT(scalar(5.0), scalar(3.0))]
    ).loss() > scalar(0.0)
    assert OR(
        [GT(scalar(3.0), scalar(5.0)), LT(scalar(5.0), scalar(3.0))]
    ).satisfy() == scalar(0.0)


# Test IFTHEN
def test_ifthen_scalars():
    assert IFTHEN(
        GT(scalar(5.0), scalar(3.0)), LT(scalar(3.0), scalar(5.0))
    ).loss() == scalar(0.0)
    assert IFTHEN(
        GT(scalar(5.0), scalar(3.0)), LT(scalar(3.0), scalar(5.0))
    ).satisfy() == scalar(1.0)
    assert IFTHEN(
        GT(scalar(3.0), scalar(5.0)), LT(scalar(3.0), scalar(5.0))
    ).loss() > scalar(0.0)
    assert IFTHEN(
        GT(scalar(3.0), scalar(5.0)), LT(scalar(3.0), scalar(5.0))
    ).satisfy() == scalar(0.0)


# Test Negate
def test_negate_scalars():
    assert NEGATE(GT(scalar(3.0), scalar(5.0))).loss() == scalar(0.0)
    assert NEGATE(GT(scalar(3.0), scalar(5.0))).satisfy() == scalar(1.0)
    assert NEGATE(GT(scalar(5.0), scalar(3.0))).loss() > scalar(0.0)
    assert NEGATE(GT(scalar(5.0), scalar(3.0))).satisfy() == scalar(0.0)
    assert NEGATE(LT(scalar(5.0), scalar(3.0))).loss() == scalar(0.0)
    assert NEGATE(LT(scalar(5.0), scalar(3.0))).satisfy() == scalar(1.0)
    assert NEGATE(LT(scalar(3.0), scalar(5.0))).loss() > scalar(0.0)
    assert NEGATE(LT(scalar(3.0), scalar(5.0))).satisfy() == scalar(0.0)
    assert NEGATE(EQ(scalar(3.0), scalar(3.0))).loss() > scalar(0.0)
    assert NEGATE(EQ(scalar(3.0), scalar(3.0))).satisfy() == scalar(0.0)
    assert NEGATE(EQ(scalar(5.0), scalar(3.0))).loss() == scalar(0.0)
    assert NEGATE(EQ(scalar(5.0), scalar(3.0))).satisfy() == scalar(1.0)
    assert NEGATE(
        AND([GT(scalar(5.0), scalar(3.0)), LT(scalar(3.0), scalar(5.0))])
    ).loss() > scalar(0.0)
    assert NEGATE(
        AND([GT(scalar(5.0), scalar(3.0)), LT(scalar(3.0), scalar(5.0))])
    ).satisfy() == scalar(0.0)
    assert NEGATE(
        OR([GT(scalar(3.0), scalar(5.0)), LT(scalar(5.0), scalar(3.0))])
    ).loss() == scalar(0.0)
    assert NEGATE(
        OR([GT(scalar(3.0), scalar(5.0)), LT(scalar(5.0), scalar(3.0))])
    ).satisfy() == scalar(1.0)
    assert NEGATE(
        IFTHEN(GT(scalar(5.0), scalar(3.0)), LT(scalar(3.0), scalar(5.0)))
    ).loss() > scalar(0.0)
    assert NEGATE(
        IFTHEN(GT(scalar(5.0), scalar(3.0)), LT(scalar(3.0), scalar(5.0)))
    ).satisfy() == scalar(0.0)
