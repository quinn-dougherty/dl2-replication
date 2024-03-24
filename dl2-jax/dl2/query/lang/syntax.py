"""Abstract syntax of the dl2 language."""
from dataclasses import dataclass
from abc import ABC
from enum import Enum
from typing import List, Optional


class Node(ABC):
    def __str__(self):
        raise NotImplementedError


class ProgramType(Enum):
    FIND = "FindProgram"
    EVAL = "EvalProgram"


@dataclass
class Program(Node):
    """
    Represents the root node of the program.

    Attributes:
        statement (ProgramType): The type of program statement (FIND or EVAL).
    """

    statement: ProgramType

    def __str__(self):
        return str(self.statement.value)


@dataclass
class EvalProgram(Node):
    """
    Represents an EVAL program statement.

    Attributes:
        exp (Expression): The expression to be evaluated.
    """

    exp: "Expression"

    def __str__(self):
        return f"EVAL {self.exp}"


@dataclass
class FindProgram(Node):
    """
    Represents a FIND program statement.

    Attributes:
        variable_declarations (VariableDeclarations): The variable declarations in the program.
        constraints (Constraints): The constraints in the program.
        variable_initialization (Optional[Initializations]): Optional variable initializations.
        return_values (Optional[Returns]): Optional return values.
    """

    variable_declarations: "VariableDeclarations"
    constraints: "Constraints"
    variable_initialization: Optional["Initializations"] = None
    return_values: Optional["Returns"] = None

    def __str__(self):
        s = f"FIND {self.variable_declarations} WHERE {self.constraints}"
        if self.variable_initialization:
            s += f" INITIALIZE {self.variable_initialization}"
        if self.return_values:
            s += f" RETURN {self.return_values}"
        return s


@dataclass
class Returns(Node):
    """
    Represents the return values of a FIND program.

    Attributes:
        values (List[Expression]): A list of expressions representing the return values.
    """

    values: List["Expression"]

    def __str__(self):
        return ", ".join(str(v) for v in self.values)


@dataclass
class Initializations(Node):
    """
    Represents variable initializations in a FIND program.

    Attributes:
        initializations (List[Initialization]): A list of variable initializations.
    """

    initializations: List["Initialization"]

    def __str__(self):
        return ", ".join(str(i) for i in self.initializations)


class InitializationType(Enum):
    CONSTANT = "Constant"
    VARIABLE = "Variable"


@dataclass
class Initialization(Node):
    """
    Represents a single variable initialization.

    Attributes:
        var (Variable): The variable being initialized.
        rhs (InitializationType): The right-hand side of the initialization (CONSTANT or VARIABLE).
    """

    var: "Variable"
    rhs: InitializationType

    def __str__(self):
        return f"{self.var} = {self.rhs.value}"


@dataclass
class VariableDeclaration(Node):
    """
    Represents a variable declaration.

    Attributes:
        identifier (str): The identifier of the variable.
        shape (Shape): The shape of the variable.
    """

    identifier: str
    shape: "Shape"

    def __str__(self):
        return f"{self.identifier}{self.shape}"


@dataclass
class VariableDeclarations(Node):
    """
    Represents a list of variable declarations.

    Attributes:
        declarations (List[VariableDeclaration]): A list of variable declarations.
    """

    declarations: List["VariableDeclaration"]

    def __str__(self):
        return ", ".join(str(d) for d in self.declarations)


@dataclass
class Constraints(Node):
    """
    Represents a set of constraints in the program.

    Attributes:
        constraints (List[Disjunction]): A list of disjunctions that make up the constraints.
    """

    constraints: List["Disjunction"]

    def __str__(self):
        return ", ".join(str(c) for c in self.constraints)


@dataclass
class Disjunction(Node):
    """
    Represents a disjunction of constraints.

    Attributes:
        c1 (Constraint): The first constraint in the disjunction.
        c2 (Optional[Constraint]): The second constraint in the disjunction (optional).
    """

    c1: "Constraint"
    c2: Optional["Constraint"] = None

    def __str__(self):
        if self.c2:
            return f"{self.c1} or {self.c2}"
        return str(self.c1)


@dataclass
class Constraint(Node):
    """
    Represents a single constraint.

    Attributes:
        is_class (bool): Indicates if the constraint is a class constraint.
        args (Optional[List[Expression]]): The arguments of the class constraint (if applicable).
        rhs (Optional[Expression]): The right-hand side of the class constraint (if applicable).
        lhs (Optional[Expression]): The left-hand side of the constraint (if not a class constraint).
        op (Optional[ConstraintOperator]): The constraint operator (if not a class constraint).
    """

    is_class: bool = False
    args: Optional[List["Expression"]] = None
    rhs: Optional["Expression"] = None
    lhs: Optional["Expression"] = None
    op: Optional["ConstraintOperator"] = None

    def __str__(self):
        if self.is_class:
            s = f'class({", ".join(str(a) for a in self.args)})'
            if self.rhs:
                s += f" = {self.rhs}"
        else:
            s = f"{self.lhs} {self.op} {self.rhs}"
        return s


@dataclass
class Expression(Node):
    """
    Represents an expression.

    Attributes:
        term (ExpressionTerm): The term in the expression.
        op (Optional[str]): The operator connecting the term to another expression (if applicable).
        exp (Optional[Expression]): The other expression connected by the operator (if applicable).
    """

    term: "ExpressionTerm"
    op: Optional[str] = None
    exp: Optional["Expression"] = None

    def __str__(self):
        if self.exp:
            return f"{self.term} {self.op} {self.exp}"
        return str(self.term)


@dataclass
class ExpressionTerm(Node):
    """
    Represents a term in an expression.

    Attributes:
        factor (ExpressionFactor): The factor in the term.
        op (Optional[str]): The operator connecting the factor to another factor (if applicable).
        rhs (Optional[ExpressionFactor]): The other factor connected by the operator (if applicable).
    """

    factor: "ExpressionFactor"
    op: Optional[str] = None
    rhs: Optional["ExpressionFactor"] = None

    def __str__(self):
        if self.rhs:
            return f"{self.factor} {self.op} {self.rhs}"
        return str(self.factor)


class ExpressionFactorType(Enum):
    OPERAND = "Operand"
    EXPRESSION = "Expression"


@dataclass
class ExpressionFactor(Node):
    """
    Represents a factor in an expression term.

    Attributes:
        function (Optional[str]): The function name (if applicable).
        args (Optional[List[Expression]]): The arguments of the function (if applicable).
        layer (Optional[str]): The layer of the function (if applicable).
        index (Optional[Index]): The index of the function (if applicable).
        op_type (Optional[ExpressionFactorType]): The type of the factor (OPERAND or EXPRESSION).
        op (Optional[Operand]): The operand (if op_type is OPERAND).
        exp (Optional[Expression]): The expression (if op_type is EXPRESSION).
    """

    function: Optional[str] = None
    args: Optional[List["Expression"]] = None
    layer: Optional[str] = None
    index: Optional["Index"] = None
    op_type: Optional[ExpressionFactorType] = None
    op: Optional["Operand"] = None
    exp: Optional["Expression"] = None

    def __str__(self):
        if self.function:
            s = f'{self.function}({", ".join(str(a) for a in self.args)})'
            if self.layer:
                s += f".{self.layer}"
            if self.index:
                s += str(self.index)
        elif self.op_type == ExpressionFactorType.OPERAND:
            s = str(self.op)
        elif self.op_type == ExpressionFactorType.EXPRESSION:
            s = f"({self.exp})"
        else:
            raise ValueError("Invalid ExpressionFactor")
        return s


@dataclass
class Variable(Node):
    """
    Represents a variable.

    Attributes:
        identifier (str): The identifier of the variable.
        index (Optional[Index]): The index of the variable (if applicable).
    """

    identifier: str
    index: Optional["Index"] = None

    def __str__(self):
        s = self.identifier
        if self.index:
            s += str(self.index)
        return s


class OperandType(Enum):
    CONSTANT = "Constant"
    VARIABLE = "Variable"
    INTERVAL = "Interval"


@dataclass
class Operand(Node):
    """
    Represents an operand.

    Attributes:
        val (OperandType): The type of the operand (CONSTANT, VARIABLE, or INTERVAL).
    """

    val: OperandType

    def __str__(self):
        return str(self.val.value)


@dataclass
class ConstraintOperator(Node):
    """
    Represents a constraint operator.

    Attributes:
        op (str): The constraint operator.
    """

    op: str

    def __str__(self):
        return self.op


@dataclass
class Operands(Node):
    """
    Represents a list of operands.

    Attributes:
        operands (List[Operand]): A list of operands.
    """

    operands: List["Operand"]

    def __str__(self):
        return ", ".join(str(o) for o in self.operands)


@dataclass
class Constant(Node):
    """
    Represents a constant value.

    Attributes:
        value (float): The constant value.
    """

    value: float

    def __str__(self):
        return str(self.value)


@dataclass
class Shape(Node):
    """
    Represents the shape of a variable.

    Attributes:
        dims (List[int]): The dimensions of the shape.
    """

    dims: List[int]

    def __str__(self):
        return f'[{", ".join(str(d) for d in self.dims)}]'


@dataclass
class Interval(Node):
    """
    Represents an interval.

    Attributes:
        start (float): The start value of the interval.
        end (float): The end value of the interval.
    """

    start: float
    end: float

    def __str__(self):
        return f"[{self.start}, {self.end}]"


class IndexType(Enum):
    STRING = "str"
    VARIABLE = "Variable"


@dataclass
class Index(Node):
    """
    Represents an index.

    Attributes:
        val (IndexType): The type of the index (STRING or VARIABLE).
    """

    val: IndexType

    def __str__(self):
        return f"[{self.val.value}]"
