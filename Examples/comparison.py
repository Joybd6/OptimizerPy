from hhgso import load_HOP
from tcheckop import load_TET
from benchmarking.benchmark import Rastrigin, Ackley
import matplotlib.pyplot as plt
ackley = Ackley()

plt.title(f"Benchmark ({ackley.name})")
eop = load_TET(ackley)
eop.run(100)

hop = load_HOP(ackley)
hop.run(100)
print(hop.history)
print(eop.history)
eop.history.name = "EHGO"
hop.history.name = "HHGSO"

plt.title(f"Benchmark ({ackley.name}")
plt.xlabel("Iteration")
plt.ylabel("Optimal Value")
eop.history.plot()
hop.history.plot()
plt.legend()
plt.show()
