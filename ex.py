from dsl import *
from dists import *

prog = Profile(Mul(Normal(1,1), Normal(1, 1)))
# prog = Profile(Normal(1, 1))
# prog = Profile(Div(Exact(2), Exact(2)))

prog = Profile(Dist(Add(Normal(1,1), Normal(1, 1)), 10))

t = 0
trials = 1000

for _ in range(trials):
     t += Profile(Mul(Normal(1,1), Normal(1, 1))).variance()

print(t / trials)







# print(prog)
# print("Estimate: ", prog.estimate())
# print("Variance: ", prog.variance())