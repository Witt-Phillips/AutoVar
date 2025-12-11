from dsl import *
from dists import *

# t = 0
# trials = 1000
# for _ in range(trials):
#      t += Normal(1, True, 1, False).variance().estimate()

# print(t / trials)

# prog = Dist(Mul(Normal(1,1), Normal(1, 1)), 1000)
prog = Dist(Mul(Normal(1, False, 1, False), Normal(1, False, 1, False)), 1000)
# prog = Dist(Normal(1, True, 1, True), 1000)

print(prog)
print("Mean: ", prog.estimate())
print("Variance: ", prog.variance().estimate())


#TODO: prog = Dist(Normal(1, True, 1, False), 1000) is outptuing 0 as the variance, but should be 1!
