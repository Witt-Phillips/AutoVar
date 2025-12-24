"""
JAX-based optimization for AutoVar.

This module provides functions for compiling programs to efficient
estimators and optimizing bias parameters.
"""

from typing import Dict, Tuple, Callable
import jax
import jax.numpy as jnp
import jax.random as jr

from .compile import to_jax, get_env_mapping


def make_estimator(
    program,
    n_samples: int = 1000
) -> Tuple[Callable, Callable, Dict[str, int]]:
    """
    Compile a program to a JAX estimator with gradient function.
    
    Args:
        program: An IntractableReal AST
        n_samples: Number of samples for Monte Carlo estimation
    
    Returns:
        Tuple of (estimate_fn, grad_fn, env_mapping) where:
        - estimate_fn(env_array, key) -> float
        - grad_fn(env_array, key) -> array of gradients
        - env_mapping: Dict mapping variable names to array indices
    """
    env_mapping = get_env_mapping(program)
    jax_fn = to_jax(program, env_mapping)
    
    def estimate_mean(env_array, key):
        keys = jr.split(key, n_samples)
        samples = jax.vmap(lambda k: jax_fn(env_array, k))(keys)
        return jnp.mean(samples)
    
    # Create gradient function w.r.t. env_array (first argument)
    grad_fn = jax.grad(estimate_mean, argnums=0)
    
    # JIT compile for speed
    estimate_jit = jax.jit(estimate_mean)
    grad_jit = jax.jit(grad_fn)
    
    return estimate_jit, grad_jit, env_mapping


def minimize_variance(
    program,
    base_env: Dict[str, float] = {},
    iterations: int = 1000,
    lr: float = 1e-2,
    lr_decay: float = 0.99,
    n_samples: int = 100,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Optimize bias parameters to minimize variance-of-variance.
    
    This is the JAX-accelerated version of variance minimization.
    
    Args:
        program: The original program (not the variance program)
        base_env: Fixed environment variables (non-bias)
        iterations: Number of optimization steps
        lr: Initial learning rate
        lr_decay: Learning rate decay per iteration
        n_samples: Samples per gradient estimate
        seed: Random seed
    
    Returns:
        Dict mapping bias variable names to optimized values
    """
    # Transform to variance-of-variance program
    variance_prog = program.variance(adaptive=True)
    var_var_prog = variance_prog.variance()
    
    # Get env mapping and compile
    env_mapping = get_env_mapping(var_var_prog)
    
    if not env_mapping:
        # No bias variables to optimize
        return {}
    
    jax_fn = to_jax(var_var_prog, env_mapping)
    
    # Identify bias variables vs fixed variables
    bias_vars = [k for k in env_mapping if k.startswith('_bias_')]
    fixed_vars = [k for k in env_mapping if not k.startswith('_bias_')]
    
    # Build initial env array
    env_array = jnp.zeros(len(env_mapping), dtype=jnp.float32)
    for name, idx in env_mapping.items():
        if name in base_env:
            env_array = env_array.at[idx].set(base_env[name])
        elif name.startswith('_bias_'):
            env_array = env_array.at[idx].set(0.5)  # Default bias
    
    # Create mask for which variables to update
    update_mask = jnp.array([
        1.0 if env_mapping.get(name, -1) == idx and name.startswith('_bias_')
        else 0.0
        for idx, name in enumerate(sorted(env_mapping.keys(), key=lambda k: env_mapping[k]))
    ])
    # Reorder to match env_mapping indices
    update_mask = jnp.zeros(len(env_mapping))
    for name, idx in env_mapping.items():
        if name.startswith('_bias_'):
            update_mask = update_mask.at[idx].set(1.0)
    
    def estimate_mean(env_array, key):
        keys = jr.split(key, n_samples)
        samples = jax.vmap(lambda k: jax_fn(env_array, k))(keys)
        return jnp.mean(samples)
    
    grad_fn = jax.grad(estimate_mean, argnums=0)
    
    # Optimization loop
    key = jr.PRNGKey(seed)
    current_lr = lr
    low_bound, high_bound = 0.02, 0.98
    
    for i in range(iterations):
        key, subkey = jr.split(key)
        grads = grad_fn(env_array, subkey)
        
        # Update only bias variables
        update = current_lr * grads * update_mask
        env_array = env_array - update
        
        # Clip bias variables to valid range
        for name, idx in env_mapping.items():
            if name.startswith('_bias_'):
                env_array = env_array.at[idx].set(
                    jnp.clip(env_array[idx], low_bound, high_bound)
                )
        
        current_lr *= lr_decay
    
    # Convert back to dict
    return {name: float(env_array[idx]) for name, idx in env_mapping.items()}


# Legacy Python-based implementation (for comparison/fallback)
def minimize_variance_python(
    program,
    base_env: Dict[str, float] = {},
    iterations: int = 1000,
    lr: float = 1e-2,
    lr_decay: float = 0.99,
) -> Dict[str, float]:
    """
    Legacy Python implementation of variance minimization.
    
    Uses the original estimate_with_grad method (if available).
    Kept for backwards compatibility and testing.
    """
    from ..utils import get_environment_dependencies
    
    variance = program.variance(env=base_env, adaptive=True)
    variance_of_variance = variance.variance(env=base_env)

    bias_vars = get_environment_dependencies(variance_of_variance)
    bias_env = {bias_var: 0.5 for bias_var in bias_vars}

    low_bound, high_bound = 0.02, 0.98

    for _ in range(1, iterations + 1):
        # This requires estimate_with_grad which may not exist in new structure
        if not hasattr(variance_of_variance, 'estimate_with_grad'):
            raise NotImplementedError(
                "Legacy minimize_variance requires estimate_with_grad method. "
                "Use the JAX-based minimize_variance instead."
            )
        
        _, grads = variance_of_variance.estimate_with_grad(dict(base_env, **bias_env))

        for v in bias_vars:
            g = float(grads.get(v, 0.0))
            new_val = bias_env[v] - lr * g
            bias_env[v] = max(low_bound, min(high_bound, new_val))

        lr *= lr_decay

    return bias_env

