import time
from typing_extensions import Self
from typing import Callable
from math import exp, log

_active_profile = None

""" IntractableReal Base Types """

class IntractableReal:
    def estimate(self) -> float:
        pass
    def variance(self) -> Self:
        pass

    def __str__(self) -> str:
        pass

class Exact(IntractableReal):
    def __init__(self, x: float):
        self.val = x
    
    def estimate(self):
        return self.val
    
    def variance(self):
        return Exact(0)
    
    def __str__(self) -> str:
        return f"Exact({self.val})"

class Dist(IntractableReal):
    def __init__(self, dist: IntractableReal, n: int):
        self.n = n
        self.dist = dist
    
    def estimate(self):
        total = 0
        for _ in range(self.n):
            total += self.dist.estimate()
        return total / self.n
    
    def variance(self):
        total = 0
        for _ in range(self.n):
            total += self.dist.variance().estimate()
        return Exact(total / self.n)
    
    def __str__(self) -> str:
        return f"Dist({self.dist}, {self.n})"

""" Profiler """
# TODO: review
class ProfileData:
    """Holds profiling data for a single Profile context."""
    def __init__(self):
        self.data = {}  # (lambda_id, sampler_uid) -> [n, total_time]
    
    def record(self, lambda_id: int, sampler_uid: int, elapsed: float):
        key = (lambda_id, sampler_uid)
        if key in self.data:
            self.data[key][0] += 1
            self.data[key][1] += elapsed
        else:
            self.data[key] = [1, elapsed]
    
    def summary(self) -> dict:
        """Returns {(lambda_id, sampler_uid): (n, avg_time)}"""
        return {k: (v[0], v[1] / v[0] if v[0] > 0 else 0) 
                for k, v in self.data.items()}


# TODO: review
class Profile(IntractableReal):
    """Wraps an IntractableReal and profiles all sampling during estimate()."""
    def __init__(self, x: IntractableReal):
        self.x = x
        self.profile_data = ProfileData()

    def estimate(self):
        global _active_profile
        saved = _active_profile           # Save existing (may be None)
        _active_profile = self.profile_data
        try:
            return self.x.estimate()
        finally:
            _active_profile = saved       # Restore previous

    def variance(self):
        return Profile(self.x.variance())  # Wrap variance too

    def __str__(self):
        return f"Profile({self.x})"

    # Convenience methods that delegate to profile_data
    def summary(self) -> dict:
        return self.profile_data.summary()


# where f is a (probabilistic, otherwise not interesting) program that returns a float
# could optionally take a list of samples to avoid running the program more than necessary
class Sampler(IntractableReal):
    _uid_counter = 0

    # TODO: review
    def __init__(self, f: Callable[[], float], known_mean: float = None, known_variance: float = None):
       self.f = f #if ((known_mean is None) or (known_variance is None)) else None # only track lambda if we need to
       self.known_mean = known_mean
       self.known_variance = known_variance
       self.uid = Sampler._uid_counter
       Sampler._uid_counter += 1
    
    # TODO: review
    def estimate(self):
        if self.known_mean is not None:
            return self.known_mean
        elif _active_profile is not None:
            start = time.perf_counter()
            result = self.f()
            elapsed = time.perf_counter() - start
            key = (id(self.f), self.uid)
            if key in _active_profile.data:
                _active_profile.data[key][0] += 1
                _active_profile.data[key][1] += elapsed
            else:
                _active_profile.data[key] = [1, elapsed]
            return result
        else:
            return self.f()

        # prev impl
        # if self.known_mean is not None:
        #     return self.known_mean
        # else:
        #     return self.f()
    
    def variance(self):
        if self.known_variance is not None:
            return Exact(self.known_variance)
        else:
            # important: we don't know the variance of this difference, so we treat it as 0?
            # diff = Exact(self.f() - self.f()) 
            # no! We should just define a 'variance sampler' and use its estimate.

            if self.known_mean is not None:
                # single sample comparison to known mean
                return Sampler(lambda: (self.f() - self.known_mean)**2)
            else:
                # two sample comparison given no known mean
                return Sampler(lambda: 0.5 * (self.f() - self.f())**2)
    
    def __str__(self) -> str:
        return f"Sampler({self.f})"
        
    
""" BinOps """

class Add(IntractableReal):
    def __init__(self, x: IntractableReal, y: IntractableReal):
        self.x = x
        self.y = y
    
    def estimate(self):
        return self.x.estimate() + self.y.estimate()
			 
    def variance(self):
        return Add(self.x.variance(), self.y.variance())
    
    def __str__(self) -> str:
        return f"Add({self.x}, {self.y})"


# syntactic sugar over Add()
class Sub(IntractableReal):
    def __init__(self, x: IntractableReal, y: IntractableReal):
        self._impl = Add(x, Mul(Exact(-1), y))
    
    def estimate(self):
        return self._impl.estimate()
    
    def variance(self):
        return self._impl.variance()
    
    def __str__(self) -> str:
        return f"{self._impl}"


class Mul(IntractableReal):
    def __init__(self, x: IntractableReal, y: IntractableReal):
        self.x = x
        self.y = y
    
    def estimate(self):
        return self.x.estimate() * self.y.estimate()
    
    # Var(XY) = Var(X)Var(Y) + Var(X)μ_Y² + Var(Y)μ_X²
    def variance(self):
        vx, vy = self.x.variance(), self.y.variance()
        ux2 = Mul(self.x, self.x) # this is not a true square, its an unbiased estimator.
        uy2 = Mul(self.y, self.y)
        return Add(
            Mul(vx, vy),
            Add(Mul(vx, uy2), Mul(vy, ux2))
        )

    def __str__(self) -> str:
        return f"Mul({self.x}, {self.y})"

# we use the delta method to approximate 1/X
# Var(1/X) ≈ Var(X) / μ_X^4

class Inv(IntractableReal):
    def __init__(self, x: IntractableReal):
        self.x = x
    
    def estimate(self):
        val = self.x.estimate()
        if abs(val) < 1e-10:
            raise ValueError("Inverting near-zero sample")

        return 1 / val
    
    def variance(self):
        vx = self.x.variance()
        ux4 = Mul(Mul(self.x, self.x), Mul(self.x, self.x))  # μ^4
        return Mul(vx, Inv(ux4))

    def __str__(self) -> str:
        return f"Inv({self.x})"


# syntactic sugar over Inv and Mul
class Div(IntractableReal):
    def __init__(self, x: IntractableReal, y: IntractableReal):
        self._impl = Mul(x, Inv(y))
    
    def estimate(self):
        return self._impl.estimate()
    
    def variance(self):
        return self._impl.variance()
    
    def __str__(self) -> str:
        return f"{self._impl}"


# critical: sometimes this value will be negative, which can violate assumptions
class Square(IntractableReal):
    def __init__(self, x: IntractableReal):
        self.x = x
        self._impl = Mul(x, x)
    
    def estimate(self):
        return self._impl.estimate()
    
    def variance(self):
        return self._impl.variance()
    
    def __str__(self) -> str:
        return f"{self._impl}"

#e^x, estiamted using the delta method
class Exp(IntractableReal):
    def __init__(self, x: IntractableReal):
        self.x = x
    
    def estimate(self):
        return exp(self.x.estimate())
    
    #Var(e^X) ≈ e^(2μ) · Var(X)
    # Wanted to do the delta method, but it was giving me a different answer than the sampler.
    # The problem is that this loses all context. We don't benefit from the fact that we know the variance of the original distribution.
    def variance(self):
        return Sampler(lambda: 0.5 * (exp(self.x.estimate()) - exp(self.x.estimate()))**2)
    
    def __str__(self) -> str:
        return f"Exp({self.x})"

class Log(IntractableReal):
    def __init__(self, x: IntractableReal):
        self.x = x
    
    def estimate(self):
        return log(self.x.estimate())
    
    def variance(self):
        return Sampler(lambda: 0.5 * (log(self.x.estimate()) - log(self.x.estimate()))**2)
    
    def __str__(self) -> str:
        return f"Log({self.x})"