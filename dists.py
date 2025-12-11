from dsl import *
import random

def Normal(mu, sigma):
    return Sampler(lambda: random.gauss(mu, sigma), known_variance=sigma**2)