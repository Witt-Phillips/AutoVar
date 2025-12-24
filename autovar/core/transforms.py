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
    
    def __str__(self) -> str:
        return f"Dist({self.dist}, {self.n})"

