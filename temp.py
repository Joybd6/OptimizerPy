from optimizers.utils import History

history0 = History()

history1 = History()
history0.add_agent(([1,2,3],5))
history1.add_agent(([1,2,3],10))

print(history0.get("fitness"))
print(history1)