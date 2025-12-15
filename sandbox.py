from src import *

# t = 0
# trials = 1000
# for _ in range(trials):
#      t += Normal(1, True, 1, False).variance().estimate()

# print(t / trials)

def coin(p: float) -> float:
    return 1 if random.random() < p else 0

prog = Sampler(NamedCallable(lambda: coin(0.5), "coin"))
prog = Profile(Dist(prog, 1000))
var = prog.variance().estimate()
est = prog.estimate()

print("Estimate: ", est)
print("Variance: ", var)
print("Summary:\n", prog.summary())