"""Binary operations for the AutoVar DSL."""

from typing import Dict
from math import exp, log

from .base import IntractableReal, Exact, Env, global_uid
import autovar.core.base as base_module


def _get_uid():
    """Get and increment the global UID counter."""
    uid = base_module.global_uid
    base_module.global_uid += 1
    return uid


class Add(IntractableReal):
    """Addition of two IntractableReal expressions."""
    
    def __init__(self, x: IntractableReal, y: IntractableReal):
        self.x = x
        self.y = y
        self._uid = _get_uid()
    
    def estimate(self, env: Dict[str, float] = {}) -> float:
        return self.x.estimate(env) + self.y.estimate(env)
    
    def variance(self, env: Dict[str, float] = {}, adaptive: bool = False) -> IntractableReal:
        from .logic import generate_biased_add
        
        if adaptive:
            return generate_biased_add(
                self.x.variance(env, adaptive),
                self.y.variance(env, adaptive),
                f"{self._uid}", env, adaptive
            )
        else:
            return Add(
                self.x.variance(env, adaptive=adaptive),
                self.y.variance(env, adaptive=adaptive)
            )
    
    def __str__(self) -> str:
        return f"Add({self.x}, {self.y})"


class Mul(IntractableReal):
    """Multiplication of two IntractableReal expressions."""
    
    def __init__(self, x: IntractableReal, y: IntractableReal):
        self.x = x
        self.y = y
        self._uid = _get_uid()
    
    def estimate(self, env: Dict[str, float] = {}) -> float:
        return self.x.estimate(env) * self.y.estimate(env)
    
    def variance(self, env: Dict[str, float] = {}, adaptive: bool = False) -> IntractableReal:
        """Var(XY) = Var(X)Var(Y) + Var(X)μ_Y² + Var(Y)μ_X²"""
        from .logic import generate_biased_add
        
        vx = self.x.variance(env, adaptive=adaptive)
        vy = self.y.variance(env, adaptive=adaptive)
        ux2 = Mul(self.x, self.x)  # Unbiased estimator of μ²
        uy2 = Mul(self.y, self.y)
        
        if adaptive:
            return generate_biased_add(
                Mul(vx, vy),
                generate_biased_add(Mul(vx, uy2), Mul(vy, ux2), f"{self._uid}_1", env, adaptive),
                f"{self._uid}_2", env, adaptive
            )
        else:
            return Add(
                Mul(vx, vy),
                Add(Mul(vx, uy2), Mul(vy, ux2))
            )
    
    def __str__(self) -> str:
        return f"Mul({self.x}, {self.y})"


class Inv(IntractableReal):
    """
    Multiplicative inverse (1/x).
    
    Only allowed with deterministic values.
    """
    
    def __init__(self, x: IntractableReal):
        from ..utils import check_deterministic
        if not check_deterministic(x):
            raise ValueError(
                f"Non-deterministic node passed to Inv, which only accepts "
                f"deterministic values. Node was {x}"
            )
        self.x = x
    
    def estimate(self, env: Dict[str, float] = {}) -> float:
        val = self.x.estimate(env)
        if abs(val) < 1e-10:
            raise ValueError("Inverting near-zero sample")
        return 1 / val
    
    def variance(self, env: Dict[str, float] = {}, adaptive: bool = False) -> Exact:
        return Exact(0)
    
    def __str__(self) -> str:
        return f"Inv({self.x})"


class Sub(IntractableReal):
    """Subtraction (syntactic sugar over Add and Mul)."""
    
    def __init__(self, x: IntractableReal, y: IntractableReal):
        self._impl = Add(x, Mul(Exact(-1), y))
    
    def estimate(self, env: Dict[str, float] = {}) -> float:
        return self._impl.estimate(env)
    
    def variance(self, env: Dict[str, float] = {}, adaptive: bool = False) -> IntractableReal:
        return self._impl.variance(env, adaptive=adaptive)
    
    def __str__(self) -> str:
        return f"Sub({self._impl.x}, {self._impl.y.y})"


class Div(IntractableReal):
    """Division (syntactic sugar over Mul and Inv)."""
    
    def __init__(self, x: IntractableReal, y: IntractableReal):
        self.x = x
        self.y = y
        self._impl = Mul(x, Inv(y))
    
    def estimate(self, env: Dict[str, float] = {}) -> float:
        return self._impl.estimate(env)
    
    def variance(self, env: Dict[str, float] = {}, adaptive: bool = False) -> IntractableReal:
        return self._impl.variance(env, adaptive=adaptive)
    
    def __str__(self) -> str:
        return f"Div({self.x}, {self.y})"


class Square(IntractableReal):
    """Square of a value (syntactic sugar over Mul)."""
    
    def __init__(self, x: IntractableReal):
        self.x = x
        self._impl = Mul(x, x)
    
    def estimate(self, env: Dict[str, float] = {}) -> float:
        return self._impl.estimate(env)
    
    def variance(self, env: Dict[str, float] = {}, adaptive: bool = False) -> IntractableReal:
        return self._impl.variance(env, adaptive=adaptive)
    
    def __str__(self) -> str:
        return f"Square({self.x})"

