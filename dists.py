from dsl import *
import random

def Normal(mu, mu_known, sigma, sigma_known):
    return Sampler(lambda: random.gauss(mu, sigma), known_mean=mu if mu_known else None, known_variance=sigma**2 if sigma_known else None)