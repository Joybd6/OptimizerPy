from hhgso import load_HOP
from tcheckop import load_TET
from benchmarking.benchmark import Rastrigin, Ackley, Step, Schwefel_1_2, Rosenbrock, Quartic_with_noise, Sphere
import matplotlib.pyplot as plt
import os
import pandas as pd

ackley = Step()

#--------------Triplecheck-----------------
#Rastringin - ok
#Sphere - ok
#Quartic_with_noise - ok
#Rosenbrock - ok
#Schwefel_1_2 - ok
#Step - ok
#Ackley - ok

#plt.title(f"Benchmark ({ackley.name}\)")
eop = load_HOP(ackley)
eop.run(10)
#eop.history.get("fitness")

#print(os.getcwd())

df = pd.DataFrame(eop.history.get("fitness"),columns=["fitness"])
df.to_csv(f"results/{ackley.name}_hhgso.csv", index=False)


#hop = load_HOP(ackley)
#hop.run(100)
#print(hop.history)
#print(eop.history)
#eop.history.name = "EHGO"
#hop.history.name = "HHGSO"
""" def plot_history(history,save_path):
    plt.title(f"Benchmark ({ackley.name})")
    plt.xlabel("Iteration")
    plt.ylabel("Optimal Value")
    for i in history:
        i.plot()
    plt.legend()
    save_path = os.path.join(save_path, f"{ackley.name}.svg")
    plt.savefig(save_path, format="svg")

plot_history([eop.history, hop.history], "../images/") """
