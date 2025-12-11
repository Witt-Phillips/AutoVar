from .dsl import *
import random

def Normal(mu, mu_known, sigma, sigma_known):
    return Sampler(lambda: random.gauss(mu, sigma), known_mean=mu if mu_known else None, known_variance=sigma**2 if sigma_known else None)

def Uniform(a, a_known, b, b_known):
    return Sampler(lambda: random.uniform(a, b), known_mean=(a+b)/2 if (a_known and b_known) else None, known_variance=((b-a)**2)/12 if (a_known and b_known) else None)
