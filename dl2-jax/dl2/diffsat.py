from config import config
from abc import ABC
from typing import Sequence
from functools import reduce
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from jax import lax


def diffsat_theta(
    a: Float[Array, "dim"], b: Float[Array, "dim"], **kwargs
) -> Float[Array, "dim"]:
    """
    Calculate the absolute difference between two arrays.

    Args:
        a (Float[Array, "dim"]): First input array.
        b (Float[Array, "dim"]): Second input array.
        **kwargs: Additional keyword arguments.

    Returns:
        Float[Array, "dim"]: Absolute difference between the input arrays.
    """
    return jnp.abs(a - b)


def diffsat_delta(
    a: Float[Array, "dim"], b: Float[Array, "dim"], **kwargs
) -> Float[Array, "dim"]:
    """
    Calculate the signed difference between two arrays.

    Args:
        a (Float[Array, "dim"]): First input array.
        b (Float[Array, "dim"]): Second input array.
        **kwargs: Additional keyword arguments.

    Returns:
        Float[Array, "dim"]: Signed difference between the input arrays.
    """
    return jnp.sign(a - b) * diffsat_theta(a, b)


class Condition(ABC):
    """
    Abstract base class for defining conditions.

    Subclasses should implement the `loss` and `satisfy` methods.
    """

    def loss(self, **kwargs):
        """
        Calculate the loss value for the condition.
        """
        pass

    def satisfy(self, **kwargs):
        """
        Check if the condition is satisfied.
        """
        pass


class BConstant(Condition):
    """
    Represents a constant boolean condition.

    Args:
        x (Float[Array, "dim"]): Constant value.
    """

    def __init__(self, x: Float[Array, "dim"]):
        self.x = x

    def loss(self) -> Float[Array, "dim"]:
        """
        Calculate the loss value for the constant condition.

        Returns:
            Float[Array, "dim"]: Loss value.
        """
        return 1.0 - self.x

    def satisfy(self) -> Float[Array, "dim"]:
        """
        Check if the constant condition is satisfied.

        Returns:
            Float[Array, "dim"]: Satisfaction value.
        """
        return self.x >= (1 - config.eps_const)


class GT(Condition):
    """
    Represents a greater than condition: a > b.

    Args:
        a (Float[Array, "dim"]): First input array.
        b (Float[Array, "dim"]): Second input array.
    """

    def __init__(self, a: Float[Array, "dim"], b: Float[Array, "dim"]):
        self.a = a
        self.b = b

    def loss(self) -> Float[Array, "dim"]:
        """
        Calculate the loss value for the greater than condition.

        Returns:
            Float[Array, "dim"]: Loss value.
        """
        if config.use_eps:
            return lax.clamp(
                min=0.0, x=diffsat_delta(self.b + config.eps, self.a), max=jnp.inf
            )
        else:
            return lax.clamp(
                min=0.0, x=diffsat_delta(self.b, self.a), max=jnp.inf
            ) + jnp.equal(self.a, self.b).astype(self.a.dtype)

    def satisfy(self) -> Float[Array, "dim"]:
        """
        Check if the greater than condition is satisfied.

        Returns:
            Float[Array, "dim"]: Satisfaction value.
        """
        return self.a > self.b + config.eps_check


class LT(Condition):
    """
    Represents a less than condition: a < b.

    Args:
        a (Float[Array, "dim"]): First input array.
        b (Float[Array, "dim"]): Second input array.
    """

    def __init__(self, a: Float[Array, "dim"], b: Float[Array, "dim"]):
        self.a = a
        self.b = b

    def loss(self) -> Float[Array, "dim"]:
        """
        Calculate the loss value for the less than condition.

        Returns:
            Float[Array, "dim"]: Loss value.
        """
        if config.use_eps:
            return lax.clamp(diffsat_delta(self.a + config.eps, self.b), min=0.0)
        else:
            return lax.clamp(diffsat_delta(self.a, self.b), min=0.0) + jnp.equal(
                self.a, self.b
            ).astype(self.a.dtype)

    def satisfy(self) -> Float[Array, "dim"]:
        """
        Check if the less than condition is satisfied.

        Returns:
            Float[Array, "dim"]: Satisfaction value.
        """
        return self.a < self.b + config.eps_check


class EQ(Condition):
    """
    Represents an equality condition: a == b.

    Args:
        a (Float[Array, "dim"]): First input array.
        b (Float[Array, "dim"]): Second input array.
    """

    def __init__(self, a: Float[Array, "dim"], b: Float[Array, "dim"]):
        self.a = a
        self.b = b

    def loss(self) -> Float[Array, "dim"]:
        """
        Calculate the loss value for the equality condition.

        Returns:
            Float[Array, "dim"]: Loss value.
        """
        return lax.clamp(diffsat_delta(self.a, self.b), min=0.0)

    def satisfy(self) -> Float[Array, "dim"]:
        """
        Check if the equality condition is satisfied.

        Returns:
            Float[Array, "dim"]: Satisfaction value.
        """
        return jnp.abs(self.a - self.b) <= config.eps_check


class GEQ(Condition):
    """
    Represents a greater than or equal to condition: a >= b.

    Args:
        a (Float[Array, "dim"]): First input array.
        b (Float[Array, "dim"]): Second input array.
    """

    def __init__(self, a: Float[Array, "dim"], b: Float[Array, "dim"]):
        self.a = a
        self.b = b

    def loss(self) -> Float[Array, "dim"]:
        """
        Calculate the loss value for the greater than or equal to condition.

        Returns:
            Float[Array, "dim"]: Loss value.
        """
        return lax.clamp(diffsat_delta(self.b, self.a), min=0.0)

    def satisfy(self) -> Float[Array, "dim"]:
        """
        Check if the greater than or equal to condition is satisfied.

        Returns:
            Float[Array, "dim"]: Satisfaction value.
        """
        return self.a + config.eps_check >= self.b


class LEQ(Condition):
    """
    Represents a less than or equal to condition: a <= b.

    Args:
        a (Float[Array, "dim"]): First input array.
        b (Float[Array, "dim"]): Second input array.
    """

    def __init__(self, a: Float[Array, "dim"], b: Float[Array, "dim"]):
        self.a = a
        self.b = b

    def loss(self) -> Float[Array, "dim"]:
        """
        Calculate the loss value for the less than or equal to condition.

        Returns:
            Float[Array, "dim"]: Loss value.
        """
        return lax.clamp(min=0.0, x=diffsat_delta(self.a, self.b), max=jnp.inf)

    def satisfy(self) -> Float[Array, "dim"]:
        """
        Check if the less than or equal to condition is satisfied.

        Returns:
            Float[Array, "dim"]: Satisfaction value.
        """
        return self.a <= self.b + config.eps_check


class AND(Condition):
    """
    Represents a logical AND condition: E1 & E2 & ... & Ek.

    Args:
        expressions (Sequence[Condition]): Sequence of conditions to be ANDed.
    """

    def __init__(self, expressions: Sequence[Condition]):
        self.expressions = expressions

    def loss(self) -> Float[Array, "dim"]:
        """
        Calculate the loss value for the AND condition.

        Returns:
            Float[Array, "dim"]: Loss value.
        """
        losses = [expr.loss() for expr in self.expressions]
        return reduce(lambda a, b: a + b, losses)

    def satisfy(self) -> Float[Array, "dim"]:
        """
        Check if the AND condition is satisfied.

        Returns:
            Float[Array, "dim"]: Satisfaction value.
        """
        ret = None
        for expr in self.expressions:
            sat = expr.satisfy()
            if ret is None:
                ret = sat.copy()
            ret *= sat
        return ret


class OR(Condition):
    """
    Represents a logical OR condition: E1 || E2 || ... || Ek.

    Args:
        expressions (Sequence[Condition]): Sequence of conditions to be ORed.
    """

    def __init__(self, expressions: Sequence[Condition]):
        self.expressions = expressions

    def loss(self) -> Float[Array, "dim"]:
        """
        Calculate the loss value for the OR condition.

        Returns:
            Float[Array, "dim"]: Loss value.
        """
        losses = [expr.loss() for expr in self.expressions]
        if config.or_ == "mul":
            return reduce(lambda a, b: a * b, losses)
        if config.or_ == "min":
            return jnp.min(jnp.stack(losses), axis=0)
        raise ValueError("Invalid or_ value in config")

    def satisfy(self) -> Float[Array, "dim"]:
        """
        Check if the OR condition is satisfied.

        Returns:
            Float[Array, "dim"]: Satisfaction value.
        """
        ret = None
        for expr in self.expressions:
            sat = expr.satisfy()
            if ret is None:
                ret = sat.copy()
            ret += sat
        return ret


class IFTHEN(Condition):
    """
    Represents an if-then condition: a -> b.

    Args:
        a (Condition): Antecedent condition.
        b (Condition): Consequent condition.
    """

    def __init__(self, a: Condition, b: Condition):
        self.a = a
        self.b = b
        self.t = OR([NEGATE(a), b])

    def loss(self) -> Float[Array, "dim"]:
        """
        Calculate the loss value for the if-then condition.

        Returns:
            Float[Array, "dim"]: Loss value.
        """
        return self.t.loss()

    def satisfy(self) -> Float[Array, "dim"]:
        """
        Check if the if-then condition is satisfied.

        Returns:
            Float[Array, "dim"]: Satisfaction value.
        """
        return self.t.satisfy()


class NEGATE(Condition):
    """
    Represents a negation condition: !E.

    Args:
        expression (Condition): Condition to be negated.
    """

    def __init__(self, expression: Condition):
        self.expression = expression

        if isinstance(expression, LT):
            self.neg = GEQ(self.expression.a, self.expression.b)
        elif isinstance(expression, GT):
            self.neg = LEQ(self.expression.a, self.expression.b)
        elif isinstance(self.expression, EQ):
            self.neg = OR(
                [
                    LT(self.expression.a, self.expression.b),
                    LT(self.expression.b, self.expression.a),
                ]
            )
        elif isinstance(self.expression, LEQ):
            self.neg = GT(self.expression.a, self.expression.b)
        elif isinstance(self.expression, GEQ):
            self.neg = LT(self.expression.a, self.expression.b)
        elif isinstance(self.expression, AND):
            neg_exprs = [NEGATE(e) for e in self.expression.expressions]
            self.neg = OR(neg_exprs)
        elif isinstance(self.expression, OR):
            neg_exprs = [NEGATE(e) for e in self.expression.expressions]
            self.neg = AND(neg_exprs)
        elif isinstance(self.expression, IFTHEN):
            self.neg = AND([self.expression.a, NEGATE(self.expression.b)])
        elif isinstance(self.expression, BConstant):
            self.neg = BConstant(1.0 - self.expression.x)
        else:
            assert False, "Class not supported %s" % str(type(expression))

    def loss(self) -> Float[Array, "dim"]:
        """
        Calculate the loss value for the negation condition.

        Returns:
            Float[Array, "dim"]: Loss value.
        """
        return self.neg.loss()

    def satisfy(self) -> Float[Array, "dim"]:
        """
        Check if the negation condition is satisfied.

        Returns:
            Float[Array, "dim"]: Satisfaction value.
        """
        return self.neg.satisfy()
