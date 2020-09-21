import pandas as pd
import numpy as np
import random
import os
import time

# index_1234 = [2, 1, 3, 4]
# temp = []
# j = 3
# for k in range(4):
#     if index_1234[k] != j:
#         temp.append(index_1234[k])
# temp.append(j)
# index_1234 = temp
# print(temp)

oil_box_info_path = 'F:\\python_workspace\\2020F\\题目\\oil_box_info.xlsx'
oil_box_data = pd.read_excel(oil_box_info_path)

q1_path = 'F:\\python_workspace\\2020F\\题目\\2020年F题--飞行器质心平衡供油策略优化\\附件5-问题4数据.xlsx'
q1_data = pd.read_excel(q1_path)
q2_data = pd.read_excel(q1_path, sheet_name='飞行器俯仰角')
U_max = [1.1, 1.8, 1.7, 1.5, 1.6, 1.1]  #输油上限(kg)
U_add = 0.01
N = q1_data.iloc[:, 1].tolist()  #需要耗油量list[7200]
V0 = [0.3, 1.5, 2.1, 1.9, 2.6, 0.8]  #初始油量(m^3)
a_up_max = 0.01  #输油上升加速度上限
a_down_max = 0.01  #输油下降加速度上限
Vm = [0.405, 1.936, 2.376, 2.652, 2.88, 1.2]  #油箱油量上限(m^3)

#V0 = [0.394, 1.875, 2.214, 2.585, 2.764, 1.195]


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
        #print(1)
        x = pow(2 * s_input * np.tan(r), 0.5)
        polygon_points.append(rotate_points[0])
        # p = x / dist(low_points[0], rotate_points[0]) * np.array(low_points[0] - rotate_points[0]) + rotate_points[0]
        # polygon_points.append(p.tolist())
        # q = p
        # q[0] += x / np.sin(r)
        # polygon_points.append(q.tolist())
        rate = pow(s_input / s, 0.5)
        p = rate * (np.array(low_points[0]) - rotate_points[0]) + rotate_points[0]
        q = rate * (np.array(low_points[1]) - rotate_points[0]) + rotate_points[0]
        polygon_points.append(p.tolist())
        polygon_points.append(q.tolist())
    elif s_input <= s_total - s:
        #print(2)
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
        #print(3)
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


def calTotalCentroid(v, alpha):
    centroidList = []
    for j in range(6):
        centroidList.append(calCentroid(j, v[j], alpha))
    s = [0, 0, 0]
    for i in range(6):
        for j in range(3):
            s[j] += 850 * v[i] * centroidList[i][j]
    return s / np.array(850 * sum(v) + 3000)


#print(calTotalCentroid(V0))

f = open(r'result3.txt', 'w')
f.flush()
v_curr = np.array(V0)  #体积
s_curr = [0] * 6  #速度
v_assume_curr = v_curr.copy()
t_start = [-60] * 6  #每个油箱开始供油的时间
flag_curr = [0] * 6  #0:油箱不运作 1:油箱运作
flag_curr_index = []
for j in range(6):
    if flag_curr[j] == 1:
        flag_curr_index.append(j)
    else:
        s_curr[j] = 0
err = []
res = []
FFF = 0
GGG = 0
HHH = 0
PPP = 0
res1 = []
for t in range(7200):
    delta = sum(s_curr) / 6
    assume_err = []
    for i in range(6):
        if i == 0:
            if v_assume_curr[1] + delta / 850 <= Vm[1]:
                v_assume_curr[1] += delta / 850
                v_assume_curr[0] -= delta / 850
            else:
                v_assume_curr[1] += Vm[1] - v_assume_curr[1]
                v_assume_curr[0] -= Vm[1] - v_assume_curr[1]
        elif i == 5:
            if v_assume_curr[4] + delta / 850 <= Vm[4]:
                v_assume_curr[4] += delta / 850
                v_assume_curr[5] -= delta / 850
            else:
                v_assume_curr[4] += Vm[4] - v_assume_curr[4]
                v_assume_curr[5] -= Vm[4] - v_assume_curr[4]
        else:
            v_assume_curr[i] -= delta / 850
        assume_err.append(sum(calTotalCentroid(v_assume_curr, q2_data.iloc[t, 1])**2))
        #2345号油箱对质心的贡献度排序(排在前面的表示对质心向理想质心移动贡献大)
        index_1234 = np.array(assume_err[1:5]).argsort() + 1
    if assume_err[0] < assume_err[5]:
        max_05 = 0  #辅助油箱16中贡献度大的下标索引
    else:
        max_05 = 5
    g = []
    count = 0
    for j in index_1234:
        if v_curr[j] < 0.0001:
            g.append(j)
            count += 1
    temp = []
    for j in index_1234:
        if j not in g:
            temp.append(j)
    index_1234 = temp + g

    for i in range(4):
        if t == 0:
            if N[t] != 0:
                s_curr[index_1234[i]] = N[t] / 2
            else:
                s_curr[index_1234[i]] = random.uniform(a_up_max / 2, a_up_max)
            flag_curr[index_1234[i]] = 1
            t_start[index_1234[i]] = 0

            flag_curr_index = []
            for j in range(6):
                if flag_curr[j] == 1:
                    flag_curr_index.append(j)
                else:
                    s_curr[j] = 0

            if i == 1:
                break
        else:
            if i < 2 and index_1234[i] in flag_curr_index:
                if N[t] != 0:
                    s_curr[index_1234[i]] = N[t] / 2 + random.uniform(a_up_max / 2, a_up_max)
                else:
                    s_curr[index_1234[i]] = random.uniform(a_up_max / 2, a_up_max)
            if i >= 2 and index_1234[i] in flag_curr_index:
                if s_curr[index_1234[i]] > a_down_max:
                    s_curr[index_1234[i]] -= a_down_max
                else:
                    if t - t_start[index_1234[i]] >= 60:
                        s_curr[index_1234[i]] = 0
                        flag_curr[index_1234[i]] = 0

                        flag_curr_index = []
                        for j in range(6):
                            if flag_curr[j] == 1:
                                flag_curr_index.append(j)
                            else:
                                s_curr[j] = 0
                        for k in range(2):
                            if index_1234[k] not in flag_curr_index:
                                if N[t] != 0:
                                    s_curr[index_1234[k]] = N[t] / 2 + random.uniform(a_up_max / 2, a_up_max)
                                else:
                                    s_curr[index_1234[k]] = random.uniform(a_up_max / 2, a_up_max)
                                flag_curr[index_1234[k]] = 1
                                t_start[index_1234[k]] = t

                                flag_curr_index = []
                                for j in range(6):
                                    if flag_curr[j] == 1:
                                        flag_curr_index.append(j)
                                    else:
                                        s_curr[j] = 0
                                break

    if t == 0:
        s_curr[max_05] = random.uniform(a_up_max / 2, a_up_max)
        flag_curr[max_05] = 1
        t_start[max_05] = t

        flag_curr_index = []
        for j in range(6):
            if flag_curr[j] == 1:
                flag_curr_index.append(j)
            else:
                s_curr[j] = 0
    else:
        if max_05 in flag_curr_index:
            if s_curr[max_05] < min(U_add + 0.5 * N[t], U_max[max_05]) - a_up_max:
                s_curr[max_05] += random.uniform(a_up_max / 2, a_up_max)
        else:
            if s_curr[5 - max_05] > a_down_max:
                s_curr[5 - max_05] -= a_down_max
            else:
                if t - t_start[5 - max_05] >= 60:
                    s_curr[5 - max_05] = 0
                    flag_curr[5 - max_05] = 0
                    if s_curr[max_05] < min(U_add + 0.5 * N[t], U_max[max_05]) - a_up_max:
                        s_curr[max_05] = random.uniform(a_up_max / 2, a_up_max)
                    flag_curr[max_05] = 1
                    t_start[max_05] = t

                    flag_curr_index = []
                    for j in range(6):
                        if flag_curr[j] == 1:
                            flag_curr_index.append(j)
                        else:
                            s_curr[j] = 0

    h = []
    for i in flag_curr_index:
        if i != 0 and i != 5:
            h.append(i)

    if v_curr[0] < s_curr[0] / 850:
        if v_curr[0] >= 0:
            v_curr[1] += v_curr[0]
        v_curr[0] = 0
    else:
        v_curr[1] += s_curr[0] / 850

    if v_curr[5] < s_curr[5] / 850:
        if v_curr[5] >= 0:
            v_curr[4] += v_curr[5]
        v_curr[5] = 0
    else:
        v_curr[4] += s_curr[5] / 850

    for j in range(6):
        if v_curr[j] < s_curr[j] / 850:
            s_curr[j] = max(v_curr[j] * 850, 0)
            v_curr[j] = 0
        else:
            v_curr[j] -= s_curr[j] / 850

    if s_curr[h[0]] + s_curr[h[1]] < N[t]:
        if v_curr[h[0]] + v_curr[h[1]] < (N[t] - s_curr[h[0]] - s_curr[h[1]]) / 850:
            pass
        else:
            if v_curr[h[0]] > (N[t] - s_curr[h[0]] - s_curr[h[1]]) / 850:
                s_curr[h[0]] += N[t] - s_curr[h[0]] - s_curr[h[1]]
                v_curr[h[0]] -= (N[t] - s_curr[h[0]] - s_curr[h[1]]) / 850
            else:
                s_curr[h[1]] += N[t] - s_curr[h[0]] - s_curr[h[1]]
                v_curr[h[1]] -= (N[t] - s_curr[h[0]] - s_curr[h[1]]) / 850
    h = []
    for i in flag_curr_index:
        if i != 0 and i != 5:
            h.append(i)
    if s_curr[h[0]] + s_curr[h[1]] < N[t]:
        if v_curr[h[0]] + v_curr[h[1]] < (N[t] - s_curr[h[0]] - s_curr[h[1]]) / 850 - 0.00000001:
            HHH += 1
            pass
    v_assume_curr = v_curr.copy()
    err.append(sum(calTotalCentroid(v_curr, q2_data.iloc[t, 1])**2))
    for j in range(6):
        f.write(str(s_curr[j]) + '\t')
        time.sleep(0.00001)
    f.write('\n')
    for j in range(6):
        if s_curr[j] < 0:
            FFF += 1
        if v_curr[j] < 0:
            GGG += 1
print(max(err))
print(FFF)
print(GGG)
print(HHH)
print(v_curr)
print(sum(v_curr))
f.close()