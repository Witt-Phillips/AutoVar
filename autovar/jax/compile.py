"""
JAX compilation for AutoVar programs.

This module provides utilities for compiling AutoVar ASTs to JAX-compatible
functions. The actual `to_jax` method is defined on each class.
"""

from typing import Dict, Callable


def get_env_mapping(node) -> Dict[str, int]:
    """
    Extract all Env variable names and create a name -> index mapping.
    
    This is used to convert between Dict-based environments and
    JAX-compatible array-based environments.
    """
    from ..utils import get_environment_dependencies
    
    env_vars = get_environment_dependencies(node)
    # Sort for deterministic ordering
    env_vars = sorted(set(env_vars))
    return {name: idx for idx, name in enumerate(env_vars)}


def to_jax(node, env_mapping: Dict[str, int]) -> Callable:
    """
    Compile an AutoVar AST to a JAX function.
    
    This is a convenience wrapper that calls node.to_jax(env_mapping).
    
    Args:
        node: An IntractableReal AST node
        env_mapping: Dict mapping Env variable names to array indices
    
    Returns:
        A function (env_array, rng_key) -> float
    """
    return node.to_jax(env_mapping)
