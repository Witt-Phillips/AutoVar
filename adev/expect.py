from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
from typing import Callable
import random
import numpy as np
import uuid

_gensym_counter = 0


def gensym(name="g"):
    """Generates a unique name."""
    global _gensym_counter
    _gensym_counter += 1
    return f"_{name}_{_gensym_counter}"


class Expr(ABC):
    pass

    @abstractmethod
    def interpret(self, env, input):
        pass

    @abstractmethod
    def expect_version(self):
        pass

    @abstractmethod
    def pretty(self):
        pass

    def __call__(self, other):
        return App(self, other)

    def __str__(self):
        return self.pretty()


# Primitives
@dataclass
class PrimitiveFunction(Expr):

    name: str
    func: Callable
    dual_impl: Expr
    expect_impl: Expr

    def interpret(self, env):
        return self.func

    def dual_version(self):
        return self.dual_impl

    def expect_version(self):
        # Assume that no estimator implementation means that this is not a function
        # dealing with intractable reals.
        if self.expect_impl is None:
            return self
        return self.expect_impl

    def pretty(self):
        return self.name


def primitive(name, dual_impl, expect_impl):
    def make_prim(f):
        return PrimitiveFunction(name, f, dual_impl, expect_impl)

    return make_prim


@dataclass
class IntractableReal:
    estimator: Callable

    def estimate(self):
        return self.estimator()


@primitive("PlusR", None, None)
def PlusR(args):
    x, y = args

    def estimate_result():
        x_hat = x.estimate()
        y_hat = y.estimate()
        return x_hat + y_hat

    return IntractableReal(estimate_result)


@primitive("TimesR", None, None)
def TimesR(args):
    x, y = args

    def estimate_result():
        x_hat = x.estimate()
        y_hat = y.estimate()
        return x_hat * y_hat

    return IntractableReal(estimate_result)


@primitive("ExpR", None, None)
def ExpR(x):

    def estimate_result():
        n = np.random.poisson(2)
        r = 1.0
        for i in range(n):
            r *= x.estimate() / 2
        return r * math.exp(2)

    return IntractableReal(estimate_result)


@primitive("CastR", None, None)
def CastR(x):

    def estimate_result():
        return x

    return IntractableReal(estimate_result)


@primitive("InfiniteSum", None, None)
def InfiniteSum(f):

    def estimate_result():
        n = np.random.geometric(0.5)
        xs = [f(i).estimate() * pow(2, i) for i in range(n)]
        return sum(xs)

    return IntractableReal(estimate_result)


# Denotes the average of the elements in a very large list;
# estimator works by computing the average of a minibatch of the given size.
@primitive("MinibatchedAverage", None, None)
def Minibatch(l, minibatch_size):

    def estimate_result():
        indices = np.random.choice(len(l), size=minibatch_size, replace=False)
        return sum([l[i] for i in indices]) / minibatch_size

    return IntractableReal(estimate_result)


@dataclass
class Distribution:

    sampler: Callable

    def sample(self):
        return self.sampler()


@primitive("FlipExpect", None, None)
def FlipExpect(p):
    def integrate(f):
        return interpret(PlusR)(
            (
                interpret(TimesR)((f(True), interpret(CastR)(p))),
                interpret(TimesR)((f(False), interpret(CastR)(1 - p))),
            )
        )

    return integrate


@primitive("Flip", None, FlipExpect)
def Flip(p):
    def sampler():
        return random.random() < p

    return Distribution(sampler)


@primitive("UniformExpect", None, None)
def UniformExpect(_):
    def integrate(f):
        return IntractableReal(lambda: f(random.random()).estimate())

    return integrate


@primitive("Uniform01", None, UniformExpect)
def Uniform01(_):
    def sampler():
        return random.random()

    return Distribution(sampler)


@primitive("NormalExpect", None, None)
def NormalExpect(mu, sigma):
    def integrate(f):
        return IntractableReal(lambda: f(np.random.normal(mu, sigma)).estimate())


@primitive("Normal", None, NormalExpect)
def Normal(mu, sigma):
    def sampler():
        # Box-Muller transform for normal distribution

        u1 = random.random()
        u2 = random.random()
        z0 = ((-2 * math.log(u1)) ** 0.5) * math.cos(2 * math.pi * u2)
        return mu + sigma * z0

    return Distribution(sampler)


@dataclass
class ConstReal(Expr):
    val: float

    def interpret(self, env):
        return self.val

    def dual_version(self):
        return Tuple(ConstReal(self.val), ConstReal(0))

    def expect_version(self):
        return self

    def pretty(self):
        return str(self.val)


@dataclass
class ConstDiscrete(Expr):
    val: any  # bool, int, etc.

    def interpret(self, env):
        return self.val

    def dual_version(self):
        return self

    def expect_version(self):
        return self

    def pretty(self):
        return str(self.val)


@dataclass
class Var(Expr):
    var: str

    def interpret(self, env):
        return env[self.var]

    def dual_version(self):
        return self

    def expect_version(self):
        return self

    def pretty(self):
        return self.var


@dataclass
class Lam(Expr):
    var: str
    body: Expr

    def interpret(self, env):
        def func(x):
            new_env = env.copy()
            new_env[self.var] = x
            return self.body.interpret(new_env)

        return func

    def dual_version(self):
        return Lam(self.var, self.body.dual_version())

    def expect_version(self):
        return Lam(self.var, self.body.expect_version())

    def pretty(self):
        return f"(lam {self.var} . {self.body.pretty()})"


@dataclass
class App(Expr):
    f: Expr
    x: Expr

    def interpret(self, env):
        f = self.f.interpret(env)
        x = self.x.interpret(env)
        return f(x)

    def dual_version(self):
        return App(self.f.dual_version(), self.x.dual_version())

    def expect_version(self):
        return App(self.f.expect_version(), self.x.expect_version())

    def pretty(self):
        if isinstance(self.f, Lam):
            return f"(let {self.f.var} = {self.x.pretty()} in {self.f.body.pretty()})"
        return f"({self.f.pretty()} {self.x.pretty()})"


def Let(var, binding, body):
    return App(Lam(var, body), binding)


@dataclass
class Tuple(Expr):
    fst: Expr
    snd: Expr

    def interpret(self, env):
        return (self.fst.interpret(env), self.snd.interpret(env))

    def dual_version(self):
        return Tuple(self.fst.dual_version(), self.snd.dual_version())

    def expect_version(self):
        return Tuple(self.fst.expect_version(), self.snd.expect_version())

    def pretty(self):
        return f"({self.fst.pretty()}, {self.snd.pretty()})"


@dataclass
class Fst(Expr):
    arg: Expr

    def interpret(self, env):
        return self.arg.interpret(env)[0]

    def dual_version(self):
        return Fst(self.arg.dual_version())

    def expect_version(self):
        return Fst(self.arg.expect_version())

    def pretty(self):
        return f"fst({self.arg.pretty()})"


@dataclass
class Snd(Expr):
    arg: Expr

    def interpret(self, env):
        return self.arg.interpret(env)[1]

    def dual_version(self):
        return Snd(self.arg.dual_version())

    def expect_version(self):
        return Snd(self.arg.expect_version())

    def pretty(self):
        return f"snd({self.arg.pretty()})"


@dataclass
class IfThenElse(Expr):
    cond: Expr
    then_expr: Expr
    else_expr: Expr

    def interpret(self, env):
        if self.cond.interpret(env):
            return self.then_expr.interpret(env)
        else:
            return self.else_expr.interpret(env)

    def dual_version(self):
        return IfThenElse(
            self.cond.dual_version(),
            self.then_expr.dual_version(),
            self.else_expr.dual_version(),
        )

    def expect_version(self):
        return IfThenElse(
            self.cond.expect_version(),
            self.then_expr.expect_version(),
            self.else_expr.expect_version(),
        )

    def pretty(self):
        return f"(if {self.cond.pretty()} then {self.then_expr.pretty()} else {self.else_expr.pretty()})"


@dataclass
class Rec(Expr):
    name: str
    var: str
    body: Expr

    def interpret(self, env):
        def func(x):
            new_env = env.copy()
            new_env[self.name] = func
            new_env[self.var] = x
            return self.body.interpret(new_env)

        return func

    def dual_version(self):
        return Rec(self.name, self.var, self.body.dual_version())

    def expect_version(self):
        return Rec(self.name, self.var, self.body.expect_version())

    def pretty(self):
        return f"(rec {self.name} {self.var} . {self.body.pretty()})"


@dataclass
class Return(Expr):

    result: Expr

    def interpret(self, env):
        val = self.result.interpret(env)
        return Distribution(lambda: val)

    def expect_version(self):
        var = gensym("integrand")
        return Lam(
            var,
            App(Var(var), self.result.expect_version()),
        )

    def pretty(self):
        return f"return {self.result.pretty()}"


@dataclass
class _Do(Expr):
    var: str
    binding: Expr
    body: Expr

    def interpret(self, env):

        mu = self.binding.interpret(env)

        def draw_sample():
            x = mu.sample()
            new_env = env.copy()
            new_env[self.var] = x
            return self.body.interpret(new_env).sample()

        return Distribution(draw_sample)

    def expect_version(self):

        integrand = gensym("integrand")
        return Lam(
            integrand,
            App(
                self.binding.expect_version(),
                Lam(self.var, App(self.body.expect_version(), Var(integrand))),
            ),
        )

    def pretty(self):
        lines = []
        curr = self
        while isinstance(curr, _Do):
            lines.append(f"  {curr.var} <- {curr.binding.pretty()};")
            curr = curr.body
        lines.append(f"  {curr.pretty()}")
        return "do {\n" + "\n".join(lines) + "\n}"


def Do(bindings, final_expr):
    expr = final_expr
    for var, binding in reversed(bindings):
        expr = _Do(var, binding, expr)
    return expr


# e.g. Do([('b', App(Flip, ConstReal(0.3))), ('x', IfThenElse(Var('b'), App(Flip, ConstReal(0.2)), App(Flip, ConstReal(0.4))))], Return(Var('x')))


def interpret(expr):
    return expr.interpret({})


PlusD = PrimitiveFunction(
    "PlusD", lambda x: (x[0][0] + x[1][0], x[0][1] + x[1][1]), None, None
)
Plus = PrimitiveFunction("Plus", lambda x: x[0] + x[1], PlusD, None)
TimesD = PrimitiveFunction(
    "TimesD",
    lambda x: (x[0][0] * x[1][0], x[0][1] * x[1][0] + x[1][1] * x[0][0]),
    None,
    None,
)
Times = PrimitiveFunction("Times", lambda x: x[0] * x[1], TimesD, None)
ExpD = PrimitiveFunction("ExpD", lambda x: math.exp(x[0]) * x[1], None, None)
Exp = PrimitiveFunction("Exp", lambda x: math.exp(x), ExpD, None)


# Assume expr : R -> R
def differentiate(expr):
    return Lam("x", Snd(App(expr.dual_version(), Tuple(Var("x"), ConstReal(1.0)))))


def run_derivative(expr, x):
    derivative = interpret(differentiate(expr))
    return derivative(x)
