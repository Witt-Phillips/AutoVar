"""
AutoVar: A DSL for variance-aware probabilistic programming.

AutoVar provides tools for building probabilistic programs and
automatically optimizing sample allocation to minimize variance.
"""

# Core language types
from .core import (
    IntractableReal,
    Exact,
    Env,
    Sampler,
    NamedCallable,
    Profile,
    ProfileData,
    Add,
    Sub,
    Mul,
    Div,
    Inv,
    Square,
    Exp,
    Log,
    If,
    generate_biased_add,
    Dist,
)

# Built-in distributions
from .distributions import (
    Normal,
    Uniform,
    Flip,
    Binomial,
    BernoulliNative,
    BinomialNative,
)

# Utilities
from .utils import (
    pretty,
    summarize,
    get_environment_dependencies,
    check_deterministic,
)

# JAX backend (optional - may not be installed)
try:
    from .jax import (
        to_jax,
        get_env_mapping,
        make_estimator,
        minimize_variance,
    )
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    # Provide stub that raises helpful error
    def minimize_variance(*args, **kwargs):
        raise ImportError(
            "JAX is required for minimize_variance. "
            "Install with: pip install jax jaxlib"
        )

__version__ = "0.2.0"

__all__ = [
    # Core types
    "IntractableReal",
    "Exact",
    "Env",
    "Sampler",
    "NamedCallable",
    "Profile",
    "ProfileData",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Inv",
    "Square",
    "Exp",
    "Log",
    "If",
    "generate_biased_add",
    "Dist",
    # Distributions
    "Normal",
    "Uniform",
    "Flip",
    "Binomial",
    "BernoulliNative",
    "BinomialNative",
    # Utilities
    "pretty",
    "summarize",
    "get_environment_dependencies",
    "check_deterministic",
    # JAX backend
    "minimize_variance",
]

# Conditional exports if JAX is available
if _HAS_JAX:
    __all__.extend([
        "to_jax",
        "get_env_mapping",
        "make_estimator",
    ])
