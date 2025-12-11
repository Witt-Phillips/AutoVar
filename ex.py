from src import *

# t = 0
# trials = 1000
# for _ in range(trials):
#      t += Normal(1, True, 1, False).variance().estimate()

# print(t / trials)

# prog = Dist(Mul(Normal(1,1), Normal(1, 1)), 1000)
# prog = Dist(Mul(Normal(1, False, 1, False), Normal(1, False, 1, False)), 1000)
# prog = Dist(Normal(1, True, 1, True), 1000)
# known = False
# prog = Dist(Mul(Dist(Uniform(0, True, 1, known), 1), Normal(1, True, 1, known)), 1000)

prog = Dist(Log(Exp(Normal(1, False, 1, False))), 10000)

print(prog)
print("Mean: ", prog.estimate())
print("Variance: ", prog.variance().estimate())