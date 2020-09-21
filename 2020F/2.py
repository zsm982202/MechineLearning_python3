import pandas as pd
import numpy as np
import random
import os
import time

oil_box_info_path = 'F:\\python_workspace\\2020F\\题目\\oil_box_info.xlsx'
oil_box_data = pd.read_excel(oil_box_info_path)

q1_path = 'F:\\python_workspace\\2020F\\题目\\2020年F题--飞行器质心平衡供油策略优化\\附件3-问题2数据.xlsx'
q1_data = pd.read_excel(q1_path)
q2_data = pd.read_excel(q1_path, sheet_name='飞行器理想质心数据')
U_max = [1.1, 1.8, 1.7, 1.5, 1.6, 1.1]  #输油上限(kg)
U_add = 0.01
N = q1_data.iloc[:, 1].tolist()  #需要耗油量list[7200]
C = np.array(q2_data.iloc[:, 1:4])  #理想质心坐标[7200*3]
V0 = [0.3, 1.5, 2.1, 1.9, 2.6, 0.8]  #初始油量(m^3)
a_up_max = 0.1  #输油上升加速度上限
a_down_max = 0.01  #输油下降加速度上限
Vm = [0.405, 1.936, 2.376, 2.652, 2.88, 1.2]  #油箱油量上限(m^3)
v_curr = np.array(V0)  #体积
s_curr = [0] * 6  #速度
v_assume_curr = v_curr.copy()
t_start = [-60] * 6  #每个油箱开始供油的时间
flag_curr = [0] * 6  #0:油箱不运作 1:油箱运作


# id:油箱编号0-5  v:当前油箱内油的体积  return:质心坐标
def calCentroid(id, v):
    oil_info = oil_box_data.iloc[id, :]
    centroid = [oil_info['x'], oil_info['y'], 0]
    s_input = v / oil_info['w']
    s_total = oil_info['l'] * oil_info['h']
    centroid[2] = oil_info['z'] - 0.5 * (s_total - s_input) / oil_info['l']
    return centroid


# v:list[6]当前时刻6个油箱内油的体积
def calTotalCentroid(v):
    oil_m = 850 * np.array(v)
    centroidList = []
    for i in range(6):
        centroidList.append(calCentroid(i, v[i]))
    #print(centroidList)
    s = [0, 0, 0]
    for i in range(6):
        for j in range(3):
            s[j] += oil_m[i] * centroidList[i][j]
    return np.array(s / np.array(sum(oil_m) + 3000))


f = open(r'result2.txt', 'w')
f.flush()

flag_curr_index = []
for j in range(6):
    if flag_curr[j] == 1:
        flag_curr_index.append(j)
    else:
        s_curr[j] = 0
err = []
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
        assume_err.append(sum((calTotalCentroid(v_assume_curr) - C[t])**2))
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
                    s_curr[index_1234[i]] = N[t] / 2
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
                                    s_curr[index_1234[k]] = N[t] / 2
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
        if not v_curr[h[0]] + v_curr[h[1]] < (N[t] - s_curr[h[0]] - s_curr[h[1]]) / 850:
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

    v_assume_curr = v_curr.copy()
    err.append(sum((calTotalCentroid(v_curr) - C[t])**2))
    for j in range(6):
        f.write(str(s_curr[j]) + '\t')
        time.sleep(0.00001)
    f.write('\n')
print(max(err))
print(v_curr)

f.close()