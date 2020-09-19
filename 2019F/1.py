import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# excel_path = 'E:\\mathematical modeling\\2019题\\2019题\\npmcm2019f\\npmcm2019-F\\附件1：数据集1-终稿.xlsx'
excel_path = 'E:\\mathematical modeling\\2019题\\2019题\\npmcm2019f\\npmcm2019-F\\附件2：数据集2-终稿.xlsx'
new_col = ['id', 'X', 'Y', 'Z', 'V', 'flag']
df = pd.read_excel(excel_path, names=new_col, skiprows=1)
df['V'][0] = 0
n = df.shape[0]
A = np.zeros([n, n])
E = []
dist_num = 0
for i in range(n):
    for j in range(n):
        A[i, j] = ((df['X'][i] - df['X'][j])**2 + (df['Y'][i] - df['Y'][j])**2 + (df['Z'][i] - df['Z'][j])**2)**0.5
        if j < n - 1:
            if df['V'][j] == 0:
                if A[i, j] > 15000:
                    A[i, j] = -1
                elif i != j:
                    E.append((i, j))
                    dist_num += 1
            else:
                if A[i, j] > 10000:
                    A[i, j] = -1
                elif i != j:
                    E.append((i, j))
                    dist_num += 1
        else:
            if A[i, j] > 20000:
                A[i, j] = -1
            elif i != j:
                E.append((i, j))
                dist_num += 1
print(n * n)
print(dist_num)
print(E)

V = []
H = []
for i in range(1, n - 1):
    if df['V'][i] == 1:
        V.append(i)
    else:
        H.append(i)
m = gp.Model("1")
x = m.addVars(n, n, vtype=GRB.BINARY)
h = m.addVars(n, vtype=GRB.CONTINUOUS, lb=0)
v = m.addVars(n, vtype=GRB.CONTINUOUS, lb=0)
# m.setObjective(gp.quicksum(x[i, j] * A[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
m.setObjective(gp.quicksum(x[i, j] * A[i, j] for (i, j) in E), GRB.MINIMIZE)

m.addConstrs(x[i, j] == 0 for i in range(n) for j in range(n) if (i, j) not in E)

m.addConstrs(gp.quicksum(x[i, j] for i in range(n) if (i, j) in E) == gp.quicksum(x[j, k] for k in range(n) if (j, k) in E) for j in range(1, n - 1))
m.addConstrs(gp.quicksum(x[i, j] for i in range(n) if (i, j) in E) <= 1 for j in range(1, n))
m.addConstrs(gp.quicksum(x[j, k] for k in range(n) if (j, k) in E) <= 1 for j in range(n - 1))
m.addConstrs(df['V'][i] * h[i] + 0.001 * A[i, j] - h[j] <= 1000000 * (1 - x[i, j]) for (i, j) in E if i != n - 1)
m.addConstrs((1 - df['V'][i]) * v[i] + 0.001 * A[i, j] - v[j] <= 1000000 * (1 - x[i, j]) for (i, j) in E if i != n - 1)
m.addConstrs(v[i] <= 20 for i in V)
m.addConstrs(h[i] <= 10 for i in V)
m.addConstrs(v[i] <= 15 for i in H)
m.addConstrs(h[i] <= 20 for i in H)
m.addConstr(h[0] == 0)
m.addConstr(v[0] == 0)
m.addConstr(v[n - 1] <= 20)
m.addConstr(h[n - 1] <= 20)
m.addConstr(gp.quicksum(x[0, j] for j in range(n) if (0, j) in E) == 1)
m.addConstr(gp.quicksum(x[j, n - 1] for j in range(n) if (j, n - 1) in E) == 1)

m.optimize()

xx = m.getAttr('x', x)
for (i, j) in E:
    if xx[i, j] == 1:
        print([i, j])
