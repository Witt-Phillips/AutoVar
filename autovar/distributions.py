"""Built-in probability distributions for AutoVar."""

import random

from .core import Sampler, NamedCallable, IntractableReal, Exact, Add


def Normal(mu: float, sigma: float, mu_known: bool = True, sigma_known: bool = True) -> IntractableReal:
    """
    Normal (Gaussian) distribution.
    
    Args:
        mu: Mean
        sigma: Standard deviation
        mu_known: Whether the mean is known (for variance computation)
        sigma_known: Whether the std dev is known (for variance computation)
    """
    callable = NamedCallable(lambda: random.gauss(mu, sigma), f"Normal({mu}, {sigma})")
    return Sampler(
        callable,
        known_mean=mu if mu_known else None,
        known_variance=sigma**2 if sigma_known else None
    )


def Uniform(a: float, b: float, a_known: bool = True, b_known: bool = True) -> IntractableReal:
    """
    Uniform distribution on [a, b].
    
    Args:
        a: Lower bound
        b: Upper bound
        a_known: Whether a is known
        b_known: Whether b is known
    """
    callable = NamedCallable(lambda: random.uniform(a, b), f"Uniform({a}, {b})")
    known = a_known and b_known
    return Sampler(
        callable,
        known_mean=(a + b) / 2 if known else None,
        known_variance=((b - a) ** 2) / 12 if known else None
    )


def Flip(p: float, p_known: bool = True) -> IntractableReal:
    """
    Bernoulli distribution (coin flip).
    
    Args:
        p: Probability of returning 1
        p_known: Whether p is known
    """
    callable = NamedCallable(lambda: 1 if random.random() < p else 0, f"Flip({p})")
    return Sampler(
        callable,
        known_mean=p if p_known else None,
        known_variance=p * (1 - p) if p_known else None
    )


def Binomial(n: int, p: float) -> IntractableReal:
    """
    Binomial distribution.
    
    Args:
        n: Number of trials
        p: Probability of success per trial
    """
    callable = NamedCallable(lambda: sum(1 for _ in range(n) if random.random() < p), f"Binomial({n}, {p})")
    return Sampler(callable)


# Native implementations (built from DSL primitives)

def BernoulliNative(p: float, p_known: bool = False) -> IntractableReal:
    """Bernoulli implemented using native sampler."""
    callable = NamedCallable(lambda: 1 if random.random() < p else 0, f"BernoulliNative({p})")
    return Sampler(
        callable,
        known_mean=p if p_known else None,
        known_variance=p * (1 - p) if p_known else None
    )


def BinomialNative(n: int, p: float) -> IntractableReal:
    """Binomial implemented as sum of Bernoulli trials."""
    terms = [BernoulliNative(p) for _ in range(n)]
    if not terms:
        return Exact(0)
    result = terms[0]
    for t in terms[1:]:
        result = Add(result, t)
    return result

