import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

oil_box_info_path = 'F:\\python_workspace\\2020F\\题目\\oil_box_info.xlsx'
oil_box_data = pd.read_excel(oil_box_info_path)


# id:油箱编号0-5  v:当前油箱内油的体积  alpha:仰俯角(-90,90)角度  return:质心坐标
def calCentroid(id, v):
    oil_info = oil_box_data.iloc[id, :]
    centroid = [oil_info['x'], oil_info['y'], 0]
    s_input = v / oil_info['w']
    s_total = oil_info['l'] * oil_info['h']
    centroid[2] = oil_info['z'] - 0.5 * (s_total - s_input) / oil_info['l']
    return centroid


q1_path = 'F:\\python_workspace\\2020F\\题目\\2020年F题--飞行器质心平衡供油策略优化\\附件3-问题2数据.xlsx'
q1_data = pd.read_excel(q1_path)
q2_data = pd.read_excel(q1_path, sheet_name='飞行器理想质心数据')
U = [1.1, 1.8, 1.7, 1.5, 1.6, 1.1]
N = q1_data.iloc[:, 1].tolist()
C = np.array(q2_data.iloc[:, 1:4])
V0 = [0.3, 1.5, 2.1, 1.9, 2.6, 0.8]
Vm = [0.405, 1.936, 2.376, 2.652, 2.88, 1.2]

NN = np.cumsum(N)


# v:list[6]当前时刻6个油箱内油的体积
def calTotalCentroid(v0, v1, v2, v3, v4, v5, t):
    #print(NN[1000])
    v = [v0, v1, v2, v3, v4, v5]
    oil_m = 850 * np.array(v)
    centroidList = []
    for i in range(6):
        centroidList.append(calCentroid(i, v[i]))
    #print(centroidList)
    s = [0, 0, 0]
    for i in range(6):
        for j in range(3):
            s[j] += oil_m[i] * centroidList[i][j]
    # return np.array(s / np.array(sum(oil_m) + 3000))
    #print(np.array(s / np.array(9.2 * 850 - NN[t] + 3000)))
    return np.array(s / np.array(9.2 * 850 - NN[t] + 3000))


# a = [1, 2, 3]
# b = [1, 2, 5]
# print(sum((np.array(a) - np.array(b)) * (np.array(a) - np.array(b))))

m = gp.Model("1")
w = m.addVars(6, 7200, vtype=GRB.BINARY)
x = m.addVars(6, 7200, vtype=GRB.CONTINUOUS, lb=0)
F = m.addVars(6, 7200, vtype=GRB.INTEGER, lb=0)
v = m.addVars(6, 7200, vtype=GRB.CONTINUOUS, lb=0)
z = m.addVar(vtype=GRB.CONTINUOUS, lb=0)
err = m.addVars(7200, 3, vtype=GRB.CONTINUOUS, lb=0)

m.setObjective(z, GRB.MINIMIZE)
# m.setObjective(
#     max(((calTotalCentroid(v[0, t], v[1, t], v[2, t], v[3, t], v[4, t], v[5, t], t) - C[t]) *
#          (calTotalCentroid(v[0, t], v[1, t], v[2, t], v[3, t], v[4, t], v[5, t], t) - C[t])).sum() for t in range(7200)), GRB.MINIMIZE)

m.addConstrs(x[i, t] <= w[i, t] * U[i] for i in range(6) for t in range(7200))
m.addConstrs(gp.quicksum(w[i, t] for i in range(1, 5)) <= 2 for t in range(7200))
m.addConstrs(gp.quicksum(w[i, t] for i in range(6)) <= 3 for t in range(7200))
m.addConstrs(x[i, t] >= N[t] for i in range(6) for t in range(7200))
m.addConstrs(v[i, 0] == V0[i] for i in range(6))
m.addConstrs(v[i, t] == v[i, t - 1] - x[i, t - 1] / 850 for i in [0, 2, 3, 5] for t in range(1, 7200))
m.addConstrs(v[1, t] == v[1, t - 1] - x[1, t - 1] / 850 + x[0, t - 1] / 850 for t in range(1, 7200))
m.addConstrs(v[4, t] == v[4, t - 1] - x[4, t - 1] / 850 + x[5, t - 1] / 850 for t in range(1, 7200))
m.addConstrs(v[i, t] <= Vm[i] for i in range(6) for t in range(7200))
m.addConstrs(F[i, t] == w[i, t] * (F[i, t - 1] + 1) for i in range(6) for t in range(1, 7200))
m.addConstrs((F[i, t - 1] - 60) * (w[i, t - 1] - w[i, t]) >= 0 for i in range(6) for t in range(1, 7200))

m.addConstrs(err[t, i] == list(calTotalCentroid(v[0, t], v[1, t], v[2, t], v[3, t], v[4, t], v[5, t], t) - C[t])[i] for i in range(3) for t in range(7200))
m.addConstrs(
    np.array(err[t, 0]) * np.array(err[t, 0]) + np.array(err[t, 1]) * np.array(err[t, 1]) + np.array(err[t, 2]) * np.array(err[t, 2]) <= z
    for t in range(7200))
# m.addConstrs(
#     sum((calTotalCentroid(v[0, t], v[1, t], v[2, t], v[3, t], v[4, t], v[5, t], t) - C[t]) *
#         (calTotalCentroid(v[0, t], v[1, t], v[2, t], v[3, t], v[4, t], v[5, t], t) - C[t])) <= z for t in range(7200))

m.optimize()

# a = calTotalCentroid(0, 0.1, 0.2, 0.3, 0.4, 0.3, 1) - C[1]
# print(sum(a * a))
