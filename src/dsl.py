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

class ProfileData:
    """Holds profiling data for a single Profile context."""
    def __init__(self):
        self.data = {}  # (lambda_name, sampler_uid) -> [n, total_time]

    def record(self, lambda_name: str, sampler_uid: int, elapsed: float):
        key = (lambda_name, sampler_uid)
        if key in self.data:
            self.data[key][0] += 1
            self.data[key][1] += elapsed
        else:
            self.data[key] = [1, elapsed]
    def merge(self, other: 'ProfileData'):
        """Merge another ProfileData's samples into this one."""
        for key, (n, total_time) in other.data.items():
            if key in self.data:
                self.data[key][0] += n
                self.data[key][1] += total_time
            else:
                self.data[key] = [n, total_time]
    
    def summary(self) -> dict:
        """Returns {(lambda_name, sampler_uid): (n, avg_time)}"""
        return ''.join(
            f"{k}: ({v[0]}, {v[1] / v[0] if v[0] > 0 else 0})\n"
            for k, v in self.data.items()
        ).lstrip()

class Profile(IntractableReal):
    """Wraps an IntractableReal and profiles all sampling during estimate()."""
    def __init__(self, x: IntractableReal):
        self.x = x
        self.profile_data = ProfileData()
        self._parent_profile = None  # Set by variance() for merging
    
    def estimate(self):
        global _active_profile
        saved = _active_profile           # Save existing (may be None)
        _active_profile = self.profile_data
        try:
            return self.x.estimate()
        finally:
            _active_profile = saved
            if self._parent_profile is not None:
                self._parent_profile.merge(self.profile_data)
                self.profile_data.data.clear()  # Avoid double-counting on next call
    
    def variance(self):
        global _active_profile
        # Enable profiling BEFORE calling x.variance() because some variance
        # implementations (like Dist.variance()) are eager and do sampling immediately.
        saved = _active_profile
        _active_profile = self.profile_data
        try:
            inner_variance = self.x.variance()
        finally:
            _active_profile = saved
        
        result = Profile(inner_variance)
        result._parent_profile = self.profile_data
        return result
    
    def __str__(self):
        return f"Profile({self.x})"
    
    # Convenience methods that delegate to profile_data
    def summary(self) -> dict:
        return self.profile_data.summary()

# where f is a (probabilistic, otherwise not interesting) program that returns a float
# could optionally take a list of samples to avoid running the program more than necessary

# Optionally, we can name lambdas for cleaner repr.
class NamedCallable:
    """Wraps a callable with a display name for cleaner repr."""
    def __init__(self, fn: Callable, name: str):
        self.fn = fn
        self.name = name
    
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
    
    def __repr__(self):
        return self.name


class Sampler(IntractableReal):
    _uid_counter = 0

    # TODO: review
    def __init__(self, f: Callable[[], float], known_mean: float = None, known_variance: float = None):
       self.f = f #if ((known_mean is None) or (known_variance is None)) else None # only track lambda if we need to
       self.known_mean = known_mean
       self.known_variance = known_variance
       self.uid = Sampler._uid_counter
       Sampler._uid_counter += 1
    
    def estimate(self):
        if self.known_mean is not None:
            return self.known_mean
        elif _active_profile is not None:
            start = time.perf_counter()
            result = self.f()
            elapsed = time.perf_counter() - start
            name = self.f.name if isinstance(self.f, NamedCallable) else repr(self.f)
            key = (name, self.uid)
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
    # TODO: review
    def variance(self):
        if self.known_variance is not None:
            return Exact(self.known_variance)
        else:
            # important: we don't know the variance of this difference, so we treat it as 0?
            # diff = Exact(self.f() - self.f()) 
            # no! We should just define a 'variance sampler' and use its estimate.

            if self.known_mean is not None:
                # single sample comparison to known mean. Cannot use estimate() because
                # it will return the known mean and diff will be 0.
                return Sampler(lambda: (self.f() - self.known_mean)**2)
                
            else:
                # two sample comparison given no known mean
                # return Sampler(lambda: 0.5 * (self.f() - self.f())**2)
                # don't like this behavior.... what if we just just did this calculation
                # ourselves to avoid another lambda call?
                e1 = self.estimate()
                e2 = self.estimate()
                return Exact((e1 - e2)**2 / 2)
                #return Sampler(lambda: 0.5 * (self.estimate() - self.estimate())**2)
    
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

""" Programming Constructs """

class If(IntractableReal):
    def __init__(self, cond: IntractableReal, if_expr: IntractableReal, else_expr: IntractableReal):
        self.cond = cond
        self.if_expr = if_expr
        self.else_expr = else_expr

        # TODO: this evaluates the condition twice. To maintain independence, we avoid caching the result.
        # Note that If statements will independently
        self._impl = Add(
            Mul(self.if_expr, self.cond),
            Mul(self.else_expr, Sub(Exact(1), self.cond))
        )
    
    def estimate(self):
        return self._impl.estimate()

    # todo: review
    def variance(self):
        # generally, we assume independence, but we can't here. Use Law of Total Variance.
        p = self.cond
        one_minus_p = Sub(Exact(1), p)
        
        # Var given independence assumed:
        # E[Var(X|C)] = p·Var(if) + (1-p)·Var(else)
        expected_var = Add(
            Mul(p, self.if_expr.variance()),
            Mul(one_minus_p, self.else_expr.variance())
        )
        
        # Compute covariance term
        # Var(E[X|C]) = p·(1-p)·(μ_if - μ_else)²
        diff = Sub(self.if_expr, self.else_expr)
        var_of_expected = Mul(Mul(p, one_minus_p), Square(diff))
        
        # Total variance
        return Add(expected_var, var_of_expected)
    
    def __str__(self) -> str:
        return f"If({self.cond}, {self.if_expr}, {self.else_expr})"