"""Mathematical functions for the AutoVar DSL."""

from typing import Dict
from math import exp, log

from .base import IntractableReal, Exact


class Exp(IntractableReal):
    """
    Exponential function e^x.
    
    Only allowed with deterministic values.
    """
    
    def __init__(self, x: IntractableReal):
        from ..utils import check_deterministic
        if not check_deterministic(x):
            raise ValueError(
                f"Non-deterministic node passed to Exp, which only accepts "
                f"deterministic values. Node was {x}"
            )
        self.x = x
    
    def estimate(self, env: Dict[str, float] = {}) -> float:
        return exp(self.x.estimate(env))
    
    def variance(self, env: Dict[str, float] = {}, adaptive: bool = False) -> Exact:
        # TODO: Could use delta method: Var(e^X) ≈ e^(2μ) · Var(X)
        return Exact(0)
    
    def to_jax(self, env_mapping):
        import jax.numpy as jnp
        inner_fn = self.x.to_jax(env_mapping)
        return lambda env_array, key: jnp.exp(inner_fn(env_array, key))
    
    def __str__(self) -> str:
        return f"Exp({self.x})"


class Log(IntractableReal):
    """
    Natural logarithm.
    
    Only allowed with deterministic values.
    """
    
    def __init__(self, x: IntractableReal):
        from ..utils import check_deterministic
        if not check_deterministic(x):
            raise ValueError(
                f"Non-deterministic node passed to Log, which only accepts "
                f"deterministic values. Node was {x}"
            )
        self.x = x
    
    def estimate(self, env: Dict[str, float] = {}) -> float:
        return log(self.x.estimate(env))
    
    def variance(self, env: Dict[str, float] = {}, adaptive: bool = False) -> Exact:
        return Exact(0)
    
    def to_jax(self, env_mapping):
        import jax.numpy as jnp
        inner_fn = self.x.to_jax(env_mapping)
        return lambda env_array, key: jnp.log(inner_fn(env_array, key))
    
    def __str__(self) -> str:
        return f"Log({self.x})"

