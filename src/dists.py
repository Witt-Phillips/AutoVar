from .dsl import *
import random

""" 
Distributions sampled from 'random' module builtins.
 """

def Normal(mu, mu_known, sigma, sigma_known) -> IntractableReal:
    callable = NamedCallable(lambda: random.gauss(mu, sigma), f"Normal({mu}, {sigma})")
    return Sampler(callable, known_mean=mu if mu_known else None, known_variance=sigma**2 if sigma_known else None)

def Uniform(a, a_known, b, b_known) -> IntractableReal:
    callable = NamedCallable(lambda: random.uniform(a, b), f"Uniform({a}, {b})")
    return Sampler(callable, known_mean=(a+b)/2 if (a_known and b_known) else None, known_variance=((b-a)**2)/12 if (a_known and b_known) else None)

# Bernoulli
def Flip(p, p_known) -> IntractableReal:
    callable = NamedCallable(lambda: 1 if random.random() < p else 0, f"Flip({p})")
    return Sampler(callable, known_mean=p if p_known else None, known_variance=p*(1-p) if p_known else None)

def Binomial(n: int, p: float) -> IntractableReal:
    callable = NamedCallable(lambda: random.binomialvariate(n, p), f"Binomial({n}, {p})")
    return Sampler(callable)