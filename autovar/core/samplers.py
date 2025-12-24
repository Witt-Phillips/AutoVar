"""Sampler types for wrapping probabilistic functions."""

import time
from typing import Callable, Dict

from .base import IntractableReal, Exact

# Global profiling context
_active_profile = None


class NamedCallable:
    """Wraps a callable with a display name for cleaner repr."""
    
    def __init__(self, fn: Callable, name: str):
        self.fn = fn
        self.name = name
    
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
    
    def __repr__(self):
        return self.name


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
    
    def summary(self) -> str:
        """Returns formatted string of profiling data."""
        return ''.join(
            f"{k}: ({v[0]}, {v[1] / v[0] if v[0] > 0 else 0})\n"
            for k, v in self.data.items()
        ).lstrip()


class Sampler(IntractableReal):
    """
    Wraps an arbitrary probabilistic function for sampling.
    
    The function f() should return a float. Optionally, known_mean and
    known_variance can be provided if the distribution parameters are known.
    """
    
    _uid_counter = 0

    def __init__(self, f: Callable[[], float], known_mean: float = None, known_variance: float = None):
        self.f = f
        self.known_mean = known_mean
        self.known_variance = known_variance
        self.uid = Sampler._uid_counter
        Sampler._uid_counter += 1
    
    def estimate(self, env: Dict[str, float] = {}) -> float:
        global _active_profile
        
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

    def variance(self, env: Dict[str, float] = {}, adaptive: bool = False) -> IntractableReal:
        if self.known_variance is not None:
            return Exact(self.known_variance)
        else:
            if self.known_mean is not None:
                # Single sample comparison to known mean
                return Sampler(
                    NamedCallable(lambda: (self.f() - self.known_mean)**2, f"{self.f}.variance")
                )
            else:
                # Two sample comparison given no known mean
                return Sampler(
                    NamedCallable(lambda: 0.5 * (self.f() - self.f())**2, f"{self.f}.variance")
                )

    def __str__(self) -> str:
        return f"Sampler({self.f})"


class Profile(IntractableReal):
    """Wraps an IntractableReal and profiles all sampling during estimate()."""
    
    def __init__(self, x: IntractableReal):
        self.x = x
        self.profile_data = ProfileData()
        self._parent_profile = None  # Set by variance() for merging
    
    def estimate(self, env: Dict[str, float] = {}) -> float:
        global _active_profile
        saved = _active_profile
        _active_profile = self.profile_data
        try:
            return self.x.estimate(env)
        finally:
            _active_profile = saved
            if self._parent_profile is not None:
                self._parent_profile.merge(self.profile_data)
                self.profile_data.data.clear()
    
    def variance(self, env: Dict[str, float] = {}, adaptive: bool = False) -> 'Profile':
        global _active_profile
        saved = _active_profile
        _active_profile = self.profile_data
        try:
            inner_variance = self.x.variance(env, adaptive)
        finally:
            _active_profile = saved
        
        result = Profile(inner_variance)
        result._parent_profile = self.profile_data
        return result
    
    def clear(self):
        self.profile_data.data.clear()
    
    def summary(self) -> str:
        return self.profile_data.summary()
    
    def __str__(self) -> str:
        return f"Profile({self.x})"

