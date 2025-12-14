from .dsl import *
import random

def Normal(mu, mu_known, sigma, sigma_known):
    return Sampler(lambda: random.gauss(mu, sigma), known_mean=mu if mu_known else None, known_variance=sigma**2 if sigma_known else None)

def Uniform(a, a_known, b, b_known):
    return Sampler(lambda: random.uniform(a, b), known_mean=(a+b)/2 if (a_known and b_known) else None, known_variance=((b-a)**2)/12 if (a_known and b_known) else None)

# Bernoulli
def Flip(p, p_known):
    return Sampler(lambda: 1 if random.random() < p else 0, known_mean=p if p_known else None, known_variance=p*(1-p) if p_known else None)