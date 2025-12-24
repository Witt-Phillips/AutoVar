"""Transform wrappers for the AutoVar DSL."""

from typing import Dict

from .base import IntractableReal, Exact
from .binops import Div


class Dist(IntractableReal):
    """
    Monte Carlo distribution wrapper.
    
    Wraps an IntractableReal and estimates its mean over n samples.
    """
    
    def __init__(self, dist: IntractableReal, n: int):
        self.n = n
        self.dist = dist
    
    def estimate(self, env: Dict[str, float] = {}) -> float:
        total = 0
        for _ in range(self.n):
            total += self.dist.estimate(env)
        return total / self.n
    
    def variance(self, env: Dict[str, float] = {}, adaptive: bool = False) -> IntractableReal:
        """Var(sample mean) = Var(X) / n"""
        inner_var = Dist(self.dist.variance(env, adaptive=adaptive), self.n)
        return Div(inner_var, Exact(self.n))
    
    def to_jax(self, env_mapping):
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        inner_fn = self.dist.to_jax(env_mapping)
        n = self.n
        
        def jax_dist(env_array, key):
            keys = jr.split(key, n)
            samples = jax.vmap(lambda k: inner_fn(env_array, k))(keys)
            return jnp.mean(samples)
        return jax_dist
    
    def __str__(self) -> str:
        return f"Dist({self.dist}, {self.n})"

