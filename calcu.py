s = 0
temp = 0
for i in range(1, 1000000):
    temp += 1 / float(i)
    s += temp / (i * (i + 1))
print(s)
