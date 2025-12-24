"""Base types for the AutoVar DSL."""

from typing import Callable, Dict, Self

# Global UID counter for unique node identification
global_uid = 0


class IntractableReal:
    """
    Base class for all AutoVar expressions.
    
    Represents a value that may be intractable to compute exactly,
    but can be estimated via sampling.
    """
    
    def estimate(self, env: Dict[str, float] = {}) -> float:
        """Estimate the value via sampling."""
        raise NotImplementedError
    
    def variance(self, env: Dict[str, float] = {}, adaptive: bool = False) -> Self:
        """
        Return a new IntractableReal that estimates the variance of this expression.
        
        If adaptive=True, inserts bias parameters for optimal sample allocation.
        """
        raise NotImplementedError
    
    def to_jax(self, env_mapping: Dict[str, int]) -> Callable:
        """Compile this node to a JAX function (env_array, key) -> float."""
        raise NotImplementedError
    
    def __str__(self) -> str:
        raise NotImplementedError


class Exact(IntractableReal):
    """A known constant value."""
    
    def __init__(self, x: float):
        self.val = x
    
    def estimate(self, env: Dict[str, float] = {}) -> float:
        return self.val
    
    def variance(self, env: Dict[str, float] = {}, adaptive: bool = False) -> 'Exact':
        return Exact(0)
    
    def to_jax(self, env_mapping: Dict[str, int]) -> Callable:
        import jax.numpy as jnp
        val = self.val
        return lambda env_array, key: jnp.float32(val)
    
    def __str__(self) -> str:
        return f"Exact({self.val})"


class Env(IntractableReal):
    """
    A named variable looked up from the environment.
    
    This is the primary mechanism for passing in bias parameters
    that control sample allocation.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def estimate(self, env: Dict[str, float] = {}) -> float:
        if self.name not in env:
            raise TypeError(f"Variable {self.name} not defined in environment")
        return env[self.name]
    
    def variance(self, env: Dict[str, float] = {}, adaptive: bool = False) -> Exact:
        return Exact(0)
    
    def to_jax(self, env_mapping: Dict[str, int]) -> Callable:
        idx = env_mapping[self.name]
        return lambda env_array, key: env_array[idx]
    
    def __str__(self) -> str:
        return f"Env({self.name})"

