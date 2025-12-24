"""
JAX compilation for AutoVar programs.

This module compiles AutoVar ASTs to JAX-compatible functions
that can be differentiated and JIT-compiled.
"""

from typing import Dict, Callable, List
import jax
import jax.numpy as jnp
import jax.random as jr


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
    
    Args:
        node: An IntractableReal AST node
        env_mapping: Dict mapping Env variable names to array indices
    
    Returns:
        A function (env_array, rng_key) -> float
    """
    from ..core import (
        Exact, Env, Sampler, If, Dist, Profile,
        Inv, Exp, Log, Add, Mul, Sub, Div, Square
    )
    
    # Exact: just return the constant
    if isinstance(node, Exact):
        val = node.val
        def jax_exact(env_array, key):
            return jnp.float32(val)
        return jax_exact
    
    # Env: look up in the array
    if isinstance(node, Env):
        idx = env_mapping[node.name]
        def jax_env(env_array, key):
            return env_array[idx]
        return jax_env
    
    # Sampler: use pure_callback to call Python function
    if isinstance(node, Sampler):
        f = node.f
        known_mean = node.known_mean
        
        def jax_sampler(env_array, key):
            if known_mean is not None:
                return jnp.float32(known_mean)
            # Call the Python function via callback
            # Note: This breaks JIT compilation but allows arbitrary Python
            result = jax.pure_callback(
                lambda: jnp.float32(f()),
                jax.ShapeDtypeStruct((), jnp.float32),
            )
            return result
        return jax_sampler
    
    # Add
    if isinstance(node, Add):
        left_fn = to_jax(node.x, env_mapping)
        right_fn = to_jax(node.y, env_mapping)
        
        def jax_add(env_array, key):
            k1, k2 = jr.split(key)
            return left_fn(env_array, k1) + right_fn(env_array, k2)
        return jax_add
    
    # Mul
    if isinstance(node, Mul):
        left_fn = to_jax(node.x, env_mapping)
        right_fn = to_jax(node.y, env_mapping)
        
        def jax_mul(env_array, key):
            k1, k2 = jr.split(key)
            return left_fn(env_array, k1) * right_fn(env_array, k2)
        return jax_mul
    
    # Sub, Div, Square: delegate to _impl
    if hasattr(node, '_impl'):
        return to_jax(node._impl, env_mapping)
    
    # Inv
    if isinstance(node, Inv):
        inner_fn = to_jax(node.x, env_mapping)
        
        def jax_inv(env_array, key):
            return 1.0 / inner_fn(env_array, key)
        return jax_inv
    
    # Exp
    if isinstance(node, Exp):
        inner_fn = to_jax(node.x, env_mapping)
        
        def jax_exp(env_array, key):
            return jnp.exp(inner_fn(env_array, key))
        return jax_exp
    
    # Log
    if isinstance(node, Log):
        inner_fn = to_jax(node.x, env_mapping)
        
        def jax_log(env_array, key):
            return jnp.log(inner_fn(env_array, key))
        return jax_log
    
    # If: soft version for differentiability
    if isinstance(node, If):
        cond_fn = to_jax(node.cond, env_mapping)
        then_fn = to_jax(node.if_expr, env_mapping)
        else_fn = to_jax(node.else_expr, env_mapping)
        
        def jax_if(env_array, key):
            k1, k2, k3 = jr.split(key, 3)
            p = cond_fn(env_array, k1)
            # Soft version: always compute both, weight by probability
            return p * then_fn(env_array, k2) + (1 - p) * else_fn(env_array, k3)
        return jax_if
    
    # Dist: vectorized Monte Carlo
    if isinstance(node, Dist):
        inner_fn = to_jax(node.dist, env_mapping)
        n = node.n
        
        def jax_dist(env_array, key):
            keys = jr.split(key, n)
            samples = jax.vmap(lambda k: inner_fn(env_array, k))(keys)
            return jnp.mean(samples)
        return jax_dist
    
    # Profile: just delegate to inner (profiling not supported in JAX)
    if isinstance(node, Profile):
        return to_jax(node.x, env_mapping)
    
    raise ValueError(f"Unknown node type for JAX compilation: {type(node)}")

