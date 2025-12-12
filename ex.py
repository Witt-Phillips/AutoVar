from src import *

# t = 0
# trials = 1000
# for _ in range(trials):
#      t += Normal(1, True, 1, False).variance().estimate()

# print(t / trials)

def coin(p: float) -> float:
    return 1 if random.random() < p else 0

prog = Sampler(lambda: coin(0.5))

prog = Profile(prog)
print(prog)
print("Mean: ", prog.estimate())
print("Variance: ", prog.variance().estimate())
print("Summary: ", prog.summary())