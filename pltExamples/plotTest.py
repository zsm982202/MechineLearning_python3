import numpy as np
import matplotlib.pyplot as plt

# X = np.linspace(0, 4 * np.pi, 1000)
# Y = np.sin(X)

# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.plot(X, Y)
# plt.show()
# #

# X = np.random.uniform(0, 1, 100)
# Y = np.random.uniform(0, 1, 100)

# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.scatter(X, Y)
# plt.show()
# #

# X = np.arange(10)
# Y = np.random.uniform(0, 1, 10)

# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.bar(X, Y)
# plt.show()
# #

# Z = np.random.uniform(0, 1, (8, 8))

# fig = plt.figure()
# ax = fig.add_subplot(111)

# #ax.imshow(Z)
# ax.contourf(Z)
# plt.show()
# #

# Z = np.random.uniform(0, 1, 4)

# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.pie(Z)
# plt.show()
# #

# mu = 100  # mean of distribution
# sigma = 15  # standard deviation of distribution
# Z = mu + sigma * np.random.randn(10000)

# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.hist(Z, rwidth=0.6, facecolor='blue')
# plt.text(20, 40, r'$\mu=0, sigma=1$')
# plt.show()
# #

X = np.arange(5)
Y = np.random.uniform(0, 1, 5)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.errorbar(X, Y, Y / 4)
plt.show()
#