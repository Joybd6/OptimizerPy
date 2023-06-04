class Data:
    data = 0
    def __init__(self):
        data = 2

li = []
data0 = Data()
data1 = Data()

li.append(data0)
li.append(data1)

for data in li:
    data.data = 52

print(li[0].data)
print(li[1].data)
print(data0.data)