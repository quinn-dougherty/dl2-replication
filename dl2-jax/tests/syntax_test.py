from dl2.query.lang.syntax import (
    Program,
    ProgramType,
    EvalProgram,
    FindProgram,
    Returns,
    Initializations,
    Initialization,
    InitializationType,
    VariableDeclaration,
    VariableDeclarations,
    Constraints,
    Disjunction,
    Constraint,
    Expression,
    ExpressionTerm,
    ExpressionFactor,
    ExpressionFactorType,
    Variable,
    OperandType,
    Operand,
    ConstraintOperator,
    Operands,
    Constant,
    Shape,
    Interval,
    IndexType,
    Index,
)


def test_program():
    exp = Expression(
        ExpressionTerm(
            ExpressionFactor(
                op_type=ExpressionFactorType.OPERAND, op=Operand(OperandType.CONSTANT)
            )
        )
    )
    eval_program = EvalProgram(exp)
    find_program = FindProgram(VariableDeclarations([]), Constraints([]))

    program1 = Program(ProgramType.EVAL)
    assert isinstance(program1.statement, ProgramType)
    assert program1.statement == ProgramType.EVAL
    assert str(program1) == "EvalProgram"

    program2 = Program(ProgramType.FIND)
    assert isinstance(program2.statement, ProgramType)
    assert program2.statement == ProgramType.FIND
    assert str(program2) == "FindProgram"


def test_eval_program():
    exp = Expression(
        ExpressionTerm(
            ExpressionFactor(
                op_type=ExpressionFactorType.OPERAND, op=Operand(OperandType.CONSTANT)
            )
        )
    )
    eval_program = EvalProgram(exp)
    assert str(eval_program) == "EVAL Constant"


def test_find_program():
    var_decls = VariableDeclarations(
        [VariableDeclaration("x", Shape([2, 3])), VariableDeclaration("y", Shape([]))]
    )
    constraints = Constraints(
        [
            Disjunction(
                Constraint(
                    lhs=Expression(
                        ExpressionTerm(
                            ExpressionFactor(
                                op_type=ExpressionFactorType.OPERAND,
                                op=Operand(OperandType.VARIABLE),
                            )
                        )
                    ),
                    op=ConstraintOperator(">="),
                    rhs=Expression(
                        ExpressionTerm(
                            ExpressionFactor(
                                op_type=ExpressionFactorType.OPERAND,
                                op=Operand(OperandType.CONSTANT),
                            )
                        )
                    ),
                )
            ),
            Disjunction(
                Constraint(
                    is_class=True,
                    args=[
                        Expression(
                            ExpressionTerm(
                                ExpressionFactor(
                                    op_type=ExpressionFactorType.OPERAND,
                                    op=Operand(OperandType.VARIABLE),
                                )
                            )
                        )
                    ],
                )
            ),
        ]
    )
    initializations = Initializations(
        [
            Initialization(Variable("x"), InitializationType.CONSTANT),
            Initialization(Variable("y"), InitializationType.VARIABLE),
        ]
    )
    returns = Returns(
        [
            Expression(
                ExpressionTerm(
                    ExpressionFactor(
                        op_type=ExpressionFactorType.OPERAND,
                        op=Operand(OperandType.VARIABLE),
                    )
                )
            ),
            Expression(
                ExpressionTerm(
                    ExpressionFactor(
                        op_type=ExpressionFactorType.OPERAND,
                        op=Operand(OperandType.VARIABLE),
                    )
                )
            ),
        ]
    )

    find_program1 = FindProgram(var_decls, constraints)
    assert (
        str(find_program1)
        == "FIND x[2, 3], y[] WHERE Variable >= Constant, class(Variable)"
    )

    find_program2 = FindProgram(var_decls, constraints, initializations)
    assert (
        str(find_program2)
        == "FIND x[2, 3], y[] WHERE Variable >= Constant, class(Variable) INITIALIZE x = Constant, y = Variable"
    )

    find_program3 = FindProgram(var_decls, constraints, initializations, returns)
    assert (
        str(find_program3)
        == "FIND x[2, 3], y[] WHERE Variable >= Constant, class(Variable) INITIALIZE x = Constant, y = Variable RETURN Variable, Variable"
    )


def test_returns():
    returns = Returns(
        [
            Expression(
                ExpressionTerm(
                    ExpressionFactor(
                        op_type=ExpressionFactorType.OPERAND,
                        op=Operand(OperandType.VARIABLE),
                    )
                )
            ),
            Expression(
                ExpressionTerm(
                    ExpressionFactor(
                        op_type=ExpressionFactorType.OPERAND,
                        op=Operand(OperandType.CONSTANT),
                    )
                )
            ),
        ]
    )
    assert str(returns) == "Variable, Constant"


def test_initializations():
    initializations = Initializations(
        [
            Initialization(Variable("x"), InitializationType.CONSTANT),
            Initialization(Variable("y"), InitializationType.VARIABLE),
        ]
    )
    assert str(initializations) == "x = Constant, y = Variable"


def test_initialization():
    initialization1 = Initialization(Variable("x"), InitializationType.CONSTANT)
    assert str(initialization1) == "x = Constant"

    initialization2 = Initialization(Variable("y"), InitializationType.VARIABLE)
    assert str(initialization2) == "y = Variable"


def test_variable_declaration():
    var_decl1 = VariableDeclaration("x", Shape([2, 3]))
    assert str(var_decl1) == "x[2, 3]"

    var_decl2 = VariableDeclaration("y", Shape([]))
    assert str(var_decl2) == "y[]"


def test_variable_declarations():
    var_decls = VariableDeclarations(
        [VariableDeclaration("x", Shape([2, 3])), VariableDeclaration("y", Shape([]))]
    )
    assert str(var_decls) == "x[2, 3], y[]"


def test_constraints():
    constraints = Constraints(
        [
            Disjunction(
                Constraint(
                    lhs=Expression(
                        ExpressionTerm(
                            ExpressionFactor(
                                op_type=ExpressionFactorType.OPERAND,
                                op=Operand(OperandType.VARIABLE),
                            )
                        )
                    ),
                    op=ConstraintOperator(">="),
                    rhs=Expression(
                        ExpressionTerm(
                            ExpressionFactor(
                                op_type=ExpressionFactorType.OPERAND,
                                op=Operand(OperandType.CONSTANT),
                            )
                        )
                    ),
                )
            ),
            Disjunction(
                Constraint(
                    is_class=True,
                    args=[
                        Expression(
                            ExpressionTerm(
                                ExpressionFactor(
                                    op_type=ExpressionFactorType.OPERAND,
                                    op=Operand(OperandType.VARIABLE),
                                )
                            )
                        )
                    ],
                )
            ),
        ]
    )
    assert str(constraints) == "Variable >= Constant, class(Variable)"


def test_disjunction():
    disjunction1 = Disjunction(
        Constraint(
            lhs=Expression(
                ExpressionTerm(
                    ExpressionFactor(
                        op_type=ExpressionFactorType.OPERAND,
                        op=Operand(OperandType.VARIABLE),
                    )
                )
            ),
            op=ConstraintOperator(">"),
            rhs=Expression(
                ExpressionTerm(
                    ExpressionFactor(
                        op_type=ExpressionFactorType.OPERAND,
                        op=Operand(OperandType.CONSTANT),
                    )
                )
            ),
        ),
        Constraint(
            lhs=Expression(
                ExpressionTerm(
                    ExpressionFactor(
                        op_type=ExpressionFactorType.OPERAND,
                        op=Operand(OperandType.VARIABLE),
                    )
                )
            ),
            op=ConstraintOperator("<"),
            rhs=Expression(
                ExpressionTerm(
                    ExpressionFactor(
                        op_type=ExpressionFactorType.OPERAND,
                        op=Operand(OperandType.CONSTANT),
                    )
                )
            ),
        ),
    )
    assert str(disjunction1) == "Variable > Constant or Variable < Constant"

    disjunction2 = Disjunction(
        Constraint(
            lhs=Expression(
                ExpressionTerm(
                    ExpressionFactor(
                        op_type=ExpressionFactorType.OPERAND,
                        op=Operand(OperandType.VARIABLE),
                    )
                )
            ),
            op=ConstraintOperator(">"),
            rhs=Expression(
                ExpressionTerm(
                    ExpressionFactor(
                        op_type=ExpressionFactorType.OPERAND,
                        op=Operand(OperandType.CONSTANT),
                    )
                )
            ),
        )
    )
    assert str(disjunction2) == "Variable > Constant"


def test_constraint():
    constraint1 = Constraint(
        lhs=Expression(
            ExpressionTerm(
                ExpressionFactor(
                    op_type=ExpressionFactorType.OPERAND,
                    op=Operand(OperandType.VARIABLE),
                )
            )
        ),
        op=ConstraintOperator(">="),
        rhs=Expression(
            ExpressionTerm(
                ExpressionFactor(
                    op_type=ExpressionFactorType.OPERAND,
                    op=Operand(OperandType.CONSTANT),
                )
            )
        ),
    )
    assert str(constraint1) == "Variable >= Constant"

    constraint2 = Constraint(
        is_class=True,
        args=[
            Expression(
                ExpressionTerm(
                    ExpressionFactor(
                        op_type=ExpressionFactorType.OPERAND,
                        op=Operand(OperandType.VARIABLE),
                    )
                )
            )
        ],
    )
    assert str(constraint2) == "class(Variable)"

    constraint3 = Constraint(
        is_class=True,
        args=[
            Expression(
                ExpressionTerm(
                    ExpressionFactor(
                        op_type=ExpressionFactorType.OPERAND,
                        op=Operand(OperandType.VARIABLE),
                    )
                )
            )
        ],
        rhs=Expression(
            ExpressionTerm(
                ExpressionFactor(
                    op_type=ExpressionFactorType.OPERAND,
                    op=Operand(OperandType.CONSTANT),
                )
            )
        ),
    )
    assert str(constraint3) == "class(Variable) = Constant"


def test_expression():
    exp1 = Expression(
        ExpressionTerm(
            ExpressionFactor(
                op_type=ExpressionFactorType.OPERAND, op=Operand(OperandType.CONSTANT)
            )
        )
    )
    assert str(exp1) == "Constant"

    exp2 = Expression(
        ExpressionTerm(
            ExpressionFactor(
                op_type=ExpressionFactorType.OPERAND, op=Operand(OperandType.VARIABLE)
            )
        ),
        op="+",
        exp=Expression(
            ExpressionTerm(
                ExpressionFactor(
                    op_type=ExpressionFactorType.OPERAND,
                    op=Operand(OperandType.CONSTANT),
                )
            )
        ),
    )
    assert str(exp2) == "Variable + Constant"


def test_expression_term():
    term1 = ExpressionTerm(
        ExpressionFactor(
            op_type=ExpressionFactorType.OPERAND, op=Operand(OperandType.CONSTANT)
        )
    )
    assert str(term1) == "Constant"

    term2 = ExpressionTerm(
        ExpressionFactor(
            op_type=ExpressionFactorType.OPERAND, op=Operand(OperandType.VARIABLE)
        ),
        op="*",
        rhs=ExpressionFactor(
            op_type=ExpressionFactorType.OPERAND, op=Operand(OperandType.CONSTANT)
        ),
    )
    assert str(term2) == "Variable * Constant"


def test_expression_factor():
    factor1 = ExpressionFactor(
        op_type=ExpressionFactorType.OPERAND, op=Operand(OperandType.CONSTANT)
    )
    assert str(factor1) == "Constant"

    factor2 = ExpressionFactor(
        op_type=ExpressionFactorType.EXPRESSION,
        exp=Expression(
            ExpressionTerm(
                ExpressionFactor(
                    op_type=ExpressionFactorType.OPERAND,
                    op=Operand(OperandType.VARIABLE),
                )
            )
        ),
    )
    assert str(factor2) == "(Variable)"

    factor3 = ExpressionFactor(
        function="foo",
        args=[
            Expression(
                ExpressionTerm(
                    ExpressionFactor(
                        op_type=ExpressionFactorType.OPERAND,
                        op=Operand(OperandType.CONSTANT),
                    )
                )
            ),
            Expression(
                ExpressionTerm(
                    ExpressionFactor(
                        op_type=ExpressionFactorType.OPERAND,
                        op=Operand(OperandType.VARIABLE),
                    )
                )
            ),
        ],
        layer="bar",
        index=Index(IndexType.STRING),
    )
    assert str(factor3) == "foo(Constant, Variable).bar[str]"


def test_variable():
    var1 = Variable("x")
    assert str(var1) == "x"

    var2 = Variable("y", Index(IndexType.VARIABLE))
    assert str(var2) == "y[Variable]"


def test_operand():
    operand1 = Operand(OperandType.CONSTANT)
    assert str(operand1) == "Constant"

    operand2 = Operand(OperandType.VARIABLE)
    assert str(operand2) == "Variable"

    operand3 = Operand(OperandType.INTERVAL)
    assert str(operand3) == "Interval"


def test_constraint_operator():
    op1 = ConstraintOperator(">=")
    assert str(op1) == ">="

    op2 = ConstraintOperator("in")
    assert str(op2) == "in"


def test_operands():
    operands = Operands(
        [
            Operand(OperandType.CONSTANT),
            Operand(OperandType.VARIABLE),
            Operand(OperandType.INTERVAL),
        ]
    )
    assert str(operands) == "Constant, Variable, Interval"


def test_constant():
    constant = Constant(42)
    assert str(constant) == "42"


def test_shape():
    shape1 = Shape([2, 3])
    assert str(shape1) == "[2, 3]"

    shape2 = Shape([])
    assert str(shape2) == "[]"


def test_interval():
    interval = Interval(0, 10)
    assert str(interval) == "[0, 10]"


def test_index():
    index1 = Index(IndexType.STRING)
    assert str(index1) == "[str]"

    index2 = Index(IndexType.VARIABLE)
    assert str(index2) == "[Variable]"
