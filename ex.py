from dsl import *
from dists import *

# t = 0
# trials = 1000

# for _ in range(trials):
#      t += Profile(Mul(Normal(1,1), Normal(1, 1))).variance()

# print(t / trials)

prog = Dist(Mul(Normal(1,1), Normal(1, 1)), 1000)
print(prog)
print("Mean: ", prog.estimate())
print("Variance: ", prog.variance().estimate())