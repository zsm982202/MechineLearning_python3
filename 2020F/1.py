import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import xlrd
import xlwt
import matplotlib.pyplot as plt

oil_box_info_path = 'F:\\python_workspace\\2020F\\题目\\oil_box_info.xlsx'
oil_box_data = pd.read_excel(oil_box_info_path)

q1_path = 'F:\\python_workspace\\2020F\\题目\\2020年F题--飞行器质心平衡供油策略优化\\附件2-问题1数据.xlsx'
q1_data = pd.read_excel(q1_path)
q2_data = pd.read_excel(q1_path, sheet_name='飞行器俯仰角')
v = oil_box_data['v0']
res = []
for i in range(q1_data.shape[0]):
    oil_m = []
    centroidList = []
    v[1] += q1_data.iloc[i, 1] / 850
    v[4] += q1_data.iloc[i, 6] / 850
    for j in range(6):
        v[j] -= q1_data.iloc[i, j + 1] / 850
        oil_m.append(850 * v[j])
        centroidList.append(calCentroid(j, v[j], q2_data.iloc[i, 1]))
    res.append(calTotalCentroid(oil_m, centroidList).tolist())

f = open(r'result1.txt', 'w')
[h, l] = np.array(res).shape
for i in range(h):
    for j in range(l):
        f.write(str(res[i][j]) + '\t')
    f.write('\n')
f.close()


def dist(x, y):
    return pow((x[0] - y[0])**2 + (x[1] - y[1])**2, 0.5)


def calPolygonCentroid(polygon_points):
    N = len(polygon_points)
    polygon_points.append(polygon_points[0])
    A = 0.5 * sum(polygon_points[i][0] * polygon_points[i + 1][1] - polygon_points[i + 1][0] * polygon_points[i][1] for i in range(N))
    Cx = 1 / (6 * A) * sum(
        (polygon_points[i][0] + polygon_points[i + 1][0]) * (polygon_points[i][0] * polygon_points[i + 1][1] - polygon_points[i + 1][0] * polygon_points[i][1])
        for i in range(N))
    Cy = 1 / (6 * A) * sum(
        (polygon_points[i][1] + polygon_points[i + 1][1]) * (polygon_points[i][0] * polygon_points[i + 1][1] - polygon_points[i + 1][0] * polygon_points[i][1])
        for i in range(N))
    return [Cx, Cy]


# id:油箱编号0-5  v:当前油箱内油的体积  alpha:仰俯角(-90,90)角度  return:质心坐标
def calCentroid(id, v, alpha):
    oil_info = oil_box_data.iloc[id, :]
    centroid = [0, oil_info['y'], 0]
    s_input = v / oil_info['w']
    r = alpha / 180 * np.pi
    rotateMat = np.mat([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    initial_points = [[oil_info['x'] - 0.5 * oil_info['l'], oil_info['z'] - 0.5 * oil_info['h']],
                      [oil_info['x'] + 0.5 * oil_info['l'], oil_info['z'] - 0.5 * oil_info['h']],
                      [oil_info['x'] - 0.5 * oil_info['l'], oil_info['z'] + 0.5 * oil_info['h']],
                      [oil_info['x'] + 0.5 * oil_info['l'], oil_info['z'] + 0.5 * oil_info['h']]]
    rotate_points = []
    for i in range(4):
        t = rotateMat * np.mat(initial_points[i]).T
        rotate_points.append(t.T.tolist())
    rotate_points = np.squeeze(rotate_points)
    rotate_points = rotate_points[rotate_points[:, 1].argsort()]
    #print(initial_points)
    #print(rotate_points)

    low_points = []
    high_points = []
    if r < 0:
        r += np.pi / 2
    d = dist(rotate_points[0], rotate_points[1])
    if r != 0 and rotate_points[1][0] < rotate_points[0][0]:
        s = 0.5 * d**2 / np.tan(r)
        low_points.append(rotate_points[1].tolist())
        low_points.append([rotate_points[1][0] + d / np.sin(r), rotate_points[1][1]])
        high_points.append([rotate_points[2][0] - d / np.sin(r), rotate_points[2][1]])
        high_points.append(rotate_points[2].tolist())
    else:
        s = 0.5 * d**2 * np.tan(r)
        low_points.append([rotate_points[1][0] - d / np.cos(r), rotate_points[1][1]])
        low_points.append(rotate_points[1].tolist())
        high_points.append(rotate_points[2].tolist())
        high_points.append([rotate_points[2][0] + d / np.cos(r), rotate_points[2][1]])
    s_total = oil_info['l'] * oil_info['h']
    polygon_points = []
    if s_input <= s:
        x = pow(2 * s_input * np.tan(r), 0.5)
        polygon_points.append(rotate_points[0])
        rate = pow(s_input / s, 0.5)
        p = rate * (np.array(low_points[0]) - rotate_points[0]) + rotate_points[0]
        q = rate * (np.array(low_points[1]) - rotate_points[0]) + rotate_points[0]
        polygon_points.append(p.tolist())
        polygon_points.append(q.tolist())
    elif s_input <= s_total - s:
        polygon_points.append(rotate_points[0].tolist())
        polygon_points.append(rotate_points[1].tolist())
        rate = (s_input - s) / (s_total - 2 * s)
        p = rate * (np.array(high_points[0]) - np.array(low_points[0])) + low_points[0]
        q = rate * (np.array(high_points[1]) - np.array(low_points[1])) + low_points[1]
        polygon_points.append(q.tolist())
        if rotate_points[1][0] < rotate_points[0][0]:
            polygon_points.append(p.tolist())
            polygon_points.append(q.tolist())
        else:
            polygon_points.append(q.tolist())
            polygon_points.append(p.tolist())
    else:
        polygon_points.append(rotate_points[0].tolist())
        polygon_points.append(rotate_points[1].tolist())
        s_up = s_total - s_input
        rate = pow(s_up / s, 0.5)
        p = rate * (np.array(high_points[0]) - rotate_points[3]) + rotate_points[3]
        q = rate * (np.array(high_points[1]) - rotate_points[3]) + rotate_points[3]
        if rotate_points[1][0] < rotate_points[0][0]:
            polygon_points.append(p.tolist())
            polygon_points.append(q.tolist())
        else:
            polygon_points.append(q.tolist())
            polygon_points.append(p.tolist())
        polygon_points.append(rotate_points[2].tolist())

    res = calPolygonCentroid(polygon_points)
    # centroid[0] = res[0] #惯性
    # centroid[2] = res[1] #惯性
    r1 = -alpha / 180 * np.pi
    rotateMat1 = np.mat([[np.cos(r1), -np.sin(r1)], [np.sin(r1), np.cos(r1)]])
    x1 = (rotateMat1 * np.mat(res).T).T.tolist()
    centroid[0] = res[0]  #飞行
    centroid[2] = res[1]
    return centroid


def calTotalCentroid(oil_m, centroidList):
    s = [0, 0, 0]
    for i in range(6):
        for j in range(3):
            s[j] += oil_m[i] * centroidList[i][j]
    return s / np.array(sum(oil_m) + 3000)