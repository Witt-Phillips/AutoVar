from typing_extensions import Self
from typing import Callable

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
    def __init__(self, dist: IntractableReal, n: float):
        self.n = n  # continuous 'allocation density'
        self.n_int = max(1, int(round(self.n)))
        self.dist = dist
    
    def estimate(self):
        total = 0
        for _ in range(self.n_int):
            total += self.dist.estimate()
        return total / self.n_int
    
    def variance(self):
        total = 0
        for _ in range(self.n_int):
            total += self.dist.variance().estimate()
        return Exact(total / self.n_int)
    
    def __str__(self) -> str:
        return f"Dist({self.dist}, {self.n})"

# where f is a (probabilistic, otherwise not interesting) program that returns a float
# could optionally take a list of samples to avoid running the program more than necessary
class Sampler(IntractableReal):
    def __init__(self, f: Callable[[], float], known_mean: float = None, known_variance: float = None):
       self.f = f if ((known_mean is None) or (known_variance is None)) else None # only track lambda if we need to
       self.known_mean = known_mean
       self.known_variance = known_variance
    
    def estimate(self):
        if self.known_mean is not None:
            return self.known_mean
        else:
            return self.f()
    
    def variance(self):
        if self.known_variance is not None:
            return Exact(self.known_variance)
        else:
            # important: we don't know the variance of this difference, so we treat it as 0?
            # diff = Exact(self.f() - self.f()) 
            # no! We should just define a 'variance sampler' and use its estimate.
            return Sampler(lambda: 0.5 * (self.f() - self.f())**2)
    
    def __str__(self) -> str:
        return f"Sampler({self.f})"
        
    

# BinOps
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