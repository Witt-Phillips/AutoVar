from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
from typing import Callable


class Expr(ABC):
    pass

    @abstractmethod
    def interpret(self, env, input):
        pass


# Primitives
@dataclass
class PrimitiveFunction(Expr):

    func: Callable

    def interpret(self, env):
        return self.func


Plus = PrimitiveFunction(lambda x: x[0] + x[1])
Times = PrimitiveFunction(lambda x: x[0] * x[1])
Exp = PrimitiveFunction(lambda x: math.exp(x))


@dataclass
class ConstReal(Expr):
    val: float

    def interpret(self, env):
        return self.val


@dataclass
class ConstDiscrete(Expr):
    val: any  # bool, int, etc.

    def interpret(self, env):
        return self.val


@dataclass
class Var(Expr):
    var: str

    def interpret(self, env):
        return env[self.var]


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


@dataclass
class App(Expr):
    f: Expr
    x: Expr

    def interpret(self, env):
        f = self.f.interpret(env)
        x = self.x.interpret(env)
        return f(x)


def Let(var, binding, body):
    return App(Lam(var, body), binding)


@dataclass
class Tuple(Expr):
    fst: Expr
    snd: Expr

    def interpret(self, env):
        return (self.fst.interpret(env), self.snd.interpret(env))


@dataclass
class Fst(Expr):
    arg: Expr

    def interpret(self, env):
        return self.arg.interpret(env)[0]


@dataclass
class Snd(Expr):
    arg: Expr

    def interpret(self, env):
        return self.arg.interpret(env)[1]


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


def interpret(expr):
    return expr.interpret({})
