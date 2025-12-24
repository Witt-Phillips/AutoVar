"""Control flow constructs for the AutoVar DSL."""

import random
from typing import Dict

from .base import IntractableReal, Exact, Env
from .binops import Add, Sub, Mul, Div, Square


class If(IntractableReal):
    """
    Conditional expression with probabilistic branching.
    
    cond should evaluate to a probability p in [0, 1].
    With probability p, evaluates if_expr; otherwise evaluates else_expr.
    """
    
    def __init__(self, cond: IntractableReal, if_expr: IntractableReal, else_expr: IntractableReal):
        self.cond = cond
        self.if_expr = if_expr
        self.else_expr = else_expr
    
    def estimate(self, env: Dict[str, float] = {}) -> float:
        p = self.cond.estimate(env)
        if random.random() < p:
            return self.if_expr.estimate(env)
        else:
            return self.else_expr.estimate(env)
    
    def variance(self, env: Dict[str, float] = {}, adaptive: bool = False) -> IntractableReal:
        """
        Uses Law of Total Variance:
        Var(X) = E[Var(X|C)] + Var(E[X|C])
        """
        p = self.cond
        one_minus_p = Sub(Exact(1), p)
        
        # E[Var(X|C)] = p·Var(if) + (1-p)·Var(else)
        expected_var = Add(
            Mul(p, self.if_expr.variance(env, adaptive=adaptive)),
            Mul(one_minus_p, self.else_expr.variance(env, adaptive=adaptive))
        )
        
        # Var(E[X|C]) = p·(1-p)·(μ_if - μ_else)²
        diff = Sub(self.if_expr, self.else_expr)
        var_of_expected = Mul(Mul(p, one_minus_p), Square(diff))
        
        # Total variance
        return Add(expected_var, var_of_expected)
    
    def to_jax(self, env_mapping):
        import jax.random as jr
        cond_fn = self.cond.to_jax(env_mapping)
        then_fn = self.if_expr.to_jax(env_mapping)
        else_fn = self.else_expr.to_jax(env_mapping)
        
        def jax_if(env_array, key):
            k1, k2, k3 = jr.split(key, 3)
            p = cond_fn(env_array, k1)
            # Soft version: always compute both, weight by probability
            return p * then_fn(env_array, k2) + (1 - p) * else_fn(env_array, k3)
        return jax_if
    
    def __str__(self) -> str:
        return f"If({self.cond}, {self.if_expr}, {self.else_expr})"


def generate_biased_add(
    left: IntractableReal,
    right: IntractableReal,
    uid: str,
    env: Dict[str, float] = {},
    adaptive: bool = False
) -> If:
    """
    Generate a biased estimator for left + right.
    
    Uses importance sampling: with probability p, evaluate left/p;
    otherwise evaluate right/(1-p). This is unbiased but the variance
    depends on p, which can be optimized.
    """
    bias_var = Env(f"_bias_{uid}")
    return If(
        bias_var,
        Div(left, bias_var),
        Div(right, Sub(Exact(1), bias_var))
    )

