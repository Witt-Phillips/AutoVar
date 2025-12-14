# Standard distributions implemented in our DSL.

from .dsl import *
import random

# the idea is that we don't use builtin distributions

def BernoulliNative(p: float, p_known: bool = False) -> IntractableReal:
    return Sampler(lambda: 1 if random.random() < p else 0, known_mean=p if p_known else None, known_variance=p*(1-p) if p_known else None)

def BinomialNative(n: int, p: float) -> IntractableReal:
    terms = [BernoulliNative(p) for _ in range(n)]
    if not terms:
        return Exact(0)
    result = terms[0]
    for t in terms[1:]:
        result = Add(result, t)
    return result

""" Proof of concept... a great extension would be to build out a more robust stdlib! """


