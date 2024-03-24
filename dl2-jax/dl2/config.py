from dataclasses import dataclass


@dataclass
class Config:
    use_eps: bool = False
    eps: float = 1e-5
    eps_const: float = 1e-5
    eps_check: float = 0
    or_: str = "mul"
    cuda: bool = False
    lr: float = 0.1
    optimizer: str = "lbfgsb"
    opt_iterations: int = 1
    use_basinhopping: bool = True
    basinhopping_t: float = 10
    basinhopping_stepsize: float = 0.1
    timeout: int = 120


config = Config()
