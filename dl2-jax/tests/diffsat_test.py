import jax.numpy as jnp
from dl2.config import config
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


# Test diffsat_theta
def test_diffsat_theta_scalars():
    assert diffsat_theta(jnp.array(5.0), jnp.array(3.0)) == jnp.array(2.0)
    assert diffsat_theta(jnp.array(3.0), jnp.array(5.0)) == jnp.array(2.0)
    assert diffsat_theta(jnp.array(3.0), jnp.array(3.0)) == jnp.array(0.0)


# Test diffsat_delta
def test_diffsat_delta_scalars():
    assert diffsat_delta(jnp.array(5.0), jnp.array(3.0)) == jnp.array(2.0)
    assert diffsat_delta(jnp.array(3.0), jnp.array(5.0)) == jnp.array(-2.0)
    assert diffsat_delta(jnp.array(3.0), jnp.array(3.0)) == jnp.array(0.0)


# Test BConstant
def test_bconstant_scalars():
    assert BConstant(jnp.array(1.0)).loss() == jnp.array(0.0)
    assert BConstant(jnp.array(1.0)).satisfy() == jnp.array(1.0)
    assert BConstant(jnp.array(0.0)).loss() == jnp.array(1.0)
    assert BConstant(jnp.array(0.0)).satisfy() == jnp.array(0.0)


# Test GT
def test_gt_scalars():
    assert GT(jnp.array(5.0), jnp.array(3.0)).loss() == jnp.array(0.0)
    assert GT(jnp.array(5.0), jnp.array(3.0)).satisfy() == jnp.array(1.0)
    assert GT(jnp.array(3.0), jnp.array(5.0)).loss() > jnp.array(0.0)
    assert GT(jnp.array(3.0), jnp.array(5.0)).satisfy() == jnp.array(0.0)
    assert GT(jnp.array(3.0), jnp.array(3.0)).loss() > jnp.array(0.0)
    assert GT(jnp.array(3.0), jnp.array(3.0)).satisfy() == jnp.array(0.0)


# Test LT
def test_lt_scalars():
    assert LT(jnp.array(3.0), jnp.array(5.0)).loss() == jnp.array(0.0)
    assert LT(jnp.array(3.0), jnp.array(5.0)).satisfy() == jnp.array(1.0)
    assert LT(jnp.array(5.0), jnp.array(3.0)).loss() > jnp.array(0.0)
    assert LT(jnp.array(5.0), jnp.array(3.0)).satisfy() == jnp.array(0.0)
    assert LT(jnp.array(3.0), jnp.array(3.0)).loss() > jnp.array(0.0)
    assert LT(jnp.array(3.0), jnp.array(3.0)).satisfy() == jnp.array(0.0)


# Test EQ
def test_eq_scalars():
    assert EQ(jnp.array(3.0), jnp.array(3.0)).loss() == jnp.array(0.0)
    assert EQ(jnp.array(3.0), jnp.array(3.0)).satisfy() == jnp.array(1.0)
    assert EQ(jnp.array(5.0), jnp.array(3.0)).loss() > jnp.array(0.0)
    assert EQ(jnp.array(5.0), jnp.array(3.0)).satisfy() == jnp.array(0.0)


# Test GEQ
def test_geq_scalars():
    assert GEQ(jnp.array(5.0), jnp.array(3.0)).loss() == jnp.array(0.0)
    assert GEQ(jnp.array(5.0), jnp.array(3.0)).satisfy() == jnp.array(1.0)
    assert GEQ(jnp.array(3.0), jnp.array(3.0)).loss() == jnp.array(0.0)
    assert GEQ(jnp.array(3.0), jnp.array(3.0)).satisfy() == jnp.array(1.0)
    assert GEQ(jnp.array(3.0), jnp.array(5.0)).loss() > jnp.array(0.0)
    assert GEQ(jnp.array(3.0), jnp.array(5.0)).satisfy() == jnp.array(0.0)


# Test LEQ
def test_leq_scalars():
    assert LEQ(jnp.array(3.0), jnp.array(5.0)).loss() == jnp.array(0.0)
    assert LEQ(jnp.array(3.0), jnp.array(5.0)).satisfy() == jnp.array(1.0)
    assert LEQ(jnp.array(3.0), jnp.array(3.0)).loss() == jnp.array(0.0)
    assert LEQ(jnp.array(3.0), jnp.array(3.0)).satisfy() == jnp.array(1.0)
    assert LEQ(jnp.array(5.0), jnp.array(3.0)).loss() > jnp.array(0.0)
    assert LEQ(jnp.array(5.0), jnp.array(3.0)).satisfy() == jnp.array(0.0)


# Test AND
def test_and_scalars():
    assert AND(
        [GT(jnp.array(5.0), jnp.array(3.0)), LT(jnp.array(3.0), jnp.array(5.0))]
    ).loss() == jnp.array(0.0)
    assert AND(
        [GT(jnp.array(5.0), jnp.array(3.0)), LT(jnp.array(3.0), jnp.array(5.0))]
    ).satisfy() == jnp.array(1.0)
    assert AND(
        [GT(jnp.array(5.0), jnp.array(3.0)), LT(jnp.array(5.0), jnp.array(3.0))]
    ).loss() > jnp.array(0.0)
    assert AND(
        [GT(jnp.array(5.0), jnp.array(3.0)), LT(jnp.array(5.0), jnp.array(3.0))]
    ).satisfy() == jnp.array(0.0)


# Test OR
def test_or_scalars():
    assert OR(
        [GT(jnp.array(5.0), jnp.array(3.0)), LT(jnp.array(3.0), jnp.array(5.0))]
    ).loss() == jnp.array(0.0)
    assert OR(
        [GT(jnp.array(5.0), jnp.array(3.0)), LT(jnp.array(3.0), jnp.array(5.0))]
    ).satisfy() == jnp.array(1.0)
    assert OR(
        [GT(jnp.array(3.0), jnp.array(5.0)), LT(jnp.array(5.0), jnp.array(3.0))]
    ).loss() > jnp.array(0.0)
    assert OR(
        [GT(jnp.array(3.0), jnp.array(5.0)), LT(jnp.array(5.0), jnp.array(3.0))]
    ).satisfy() == jnp.array(0.0)


# Test IFTHEN
def test_ifthen_scalars():
    assert IFTHEN(
        GT(jnp.array(5.0), jnp.array(3.0)), LT(jnp.array(3.0), jnp.array(5.0))
    ).loss() == jnp.array(0.0)
    assert IFTHEN(
        GT(jnp.array(5.0), jnp.array(3.0)), LT(jnp.array(3.0), jnp.array(5.0))
    ).satisfy() == jnp.array(1.0)
    assert IFTHEN(
        GT(jnp.array(3.0), jnp.array(5.0)), LT(jnp.array(3.0), jnp.array(5.0))
    ).loss() == jnp.array(0.0)
    assert IFTHEN(
        GT(jnp.array(3.0), jnp.array(5.0)), LT(jnp.array(3.0), jnp.array(5.0))
    ).satisfy() == jnp.array(1.0)


# Test Negate
def test_negate_scalars():
    assert NEGATE(GT(jnp.array(3.0), jnp.array(5.0))).loss() == jnp.array(0.0)
    assert NEGATE(GT(jnp.array(3.0), jnp.array(5.0))).satisfy() == jnp.array(1.0)
    assert NEGATE(GT(jnp.array(5.0), jnp.array(3.0))).loss() > jnp.array(0.0)
    assert NEGATE(GT(jnp.array(5.0), jnp.array(3.0))).satisfy() == jnp.array(0.0)
    assert NEGATE(LT(jnp.array(5.0), jnp.array(3.0))).loss() == jnp.array(0.0)
    assert NEGATE(LT(jnp.array(5.0), jnp.array(3.0))).satisfy() == jnp.array(1.0)
    assert NEGATE(LT(jnp.array(3.0), jnp.array(5.0))).loss() > jnp.array(0.0)
    assert NEGATE(LT(jnp.array(3.0), jnp.array(5.0))).satisfy() == jnp.array(0.0)
    assert NEGATE(EQ(jnp.array(3.0), jnp.array(3.0))).loss() > jnp.array(0.0)
    assert NEGATE(EQ(jnp.array(3.0), jnp.array(3.0))).satisfy() == jnp.array(0.0)
    assert NEGATE(EQ(jnp.array(5.0), jnp.array(3.0))).loss() == jnp.array(0.0)
    assert NEGATE(EQ(jnp.array(5.0), jnp.array(3.0))).satisfy() == jnp.array(1.0)
    assert NEGATE(
        AND([GT(jnp.array(5.0), jnp.array(3.0)), LT(jnp.array(3.0), jnp.array(5.0))])
    ).loss() > jnp.array(0.0)
    assert NEGATE(
        AND([GT(jnp.array(5.0), jnp.array(3.0)), LT(jnp.array(3.0), jnp.array(5.0))])
    ).satisfy() == jnp.array(0.0)
    assert NEGATE(
        OR([GT(jnp.array(3.0), jnp.array(5.0)), LT(jnp.array(5.0), jnp.array(3.0))])
    ).loss() == jnp.array(0.0)
    assert NEGATE(
        OR([GT(jnp.array(3.0), jnp.array(5.0)), LT(jnp.array(5.0), jnp.array(3.0))])
    ).satisfy() == jnp.array(1.0)
    assert NEGATE(
        IFTHEN(GT(jnp.array(5.0), jnp.array(3.0)), LT(jnp.array(3.0), jnp.array(5.0)))
    ).loss() > jnp.array(0.0)
    assert NEGATE(
        IFTHEN(GT(jnp.array(5.0), jnp.array(3.0)), LT(jnp.array(3.0), jnp.array(5.0)))
    ).satisfy() == jnp.array(0.0)
