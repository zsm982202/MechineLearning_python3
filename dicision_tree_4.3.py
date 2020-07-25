"""
Author: zsm
Created on: 2020.5.26 18:30
"""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from math import log2


#处理连续值
def process_continuous_attr(col_index, column_data, data_set):
    sorted_con_attr = sorted(column_data)
    #cut_points: 划分点集合
    cut_points = [round((sorted_con_attr[i] + sorted_con_attr[i + 1]) / 2, 3) for i in range(len(sorted_con_attr) - 1)]

    data_set[data_set.columns[col_index]] = data_set[data_set.columns[col_index]].apply(str)
    temp_data = data_set.copy(deep=True)
    #最大划分点信息增益
    max_cut_gain = 0
    #最佳划分点
    best_cut_points = cut_points[0]
    for cut in cut_points:
        #temp_data[data_set.columns[col_index]] = str(data_set.columns[col_index]) + '<=' + str(cut)
        for i in range(len(column_data)):
            if float(temp_data[data_set.columns[col_index]][i]) <= cut:
                temp_data.loc[i, data_set.columns[col_index]] = '是'
            else:
                temp_data.loc[i, data_set.columns[col_index]] = '否'
        if calcu_each_gain(temp_data[data_set.columns[col_index]], temp_data) > max_cut_gain:
            max_cut_gain = calcu_each_gain(temp_data[data_set.columns[col_index]], temp_data)
            best_cut_points = cut
        temp_data = data_set.copy(deep=True)
    for i in range(len(column_data)):
        if float(temp_data[data_set.columns[col_index]][i]) <= best_cut_points:
            temp_data.loc[i, data_set.columns[col_index]] = '是'
        else:
            temp_data.loc[i, data_set.columns[col_index]] = '否'
    #columns.name: 密度->密度<=0.381
    temp_data.rename(columns={temp_data.columns[col_index]: str(data_set.columns[col_index]) + '<=' + str(best_cut_points)}, inplace=True)
    return temp_data


# 获得出现最多的label
def get_most_label(label_list):
    #label_dict: {'是':8, '否':9}
    label_dict = {}
    for i in label_list:
        label_dict[i] = label_dict.get(i, 0) + 1
    #key:按dict第二个元素排序,reverse=Ture表示降序
    dec_label_dict = sorted(label_dict.items(), key=lambda x: x[1], reverse=True)
    #label_dict: {'否':9, '是':8}
    return dec_label_dict[0][0]


#data:np.array with shape [m, d]. Input.
#label_dict: {'是':8, '否':9}
#m: data行数
def get_counts(label_list):
    label_len = len(label_list)
    label_dict = {}
    for d in label_list:
        label_dict[d] = label_dict.get(d, 0) + 1
    return label_dict, label_len


# 计算信息熵
def calcu_entropy(label_list):
    label_dict, label_len = get_counts(label_list)
    ent = -sum([1.0 * v / label_len * log2(v / label_len) for v in label_dict.values()])
    return ent


# 计算每个feature的信息增益
#column_data: np.array with shape [m, 1]. Input.
#temp_data: np.array with shape [m, d]. Label.
def calcu_each_gain(column_data, temp_data):
    label_list = temp_data.iloc[:, -1]
    label_len = len(label_list)
    grouped = label_list.groupby(by=column_data)
    temp = sum([len(g[1]) / label_len * calcu_entropy(g[1]) for g in list(grouped)])
    return calcu_entropy(label_list) - temp


def get_max_gain(temp_data):
    columns_entropy = [(col, calcu_each_gain(temp_data[col], temp_data)) for col in temp_data.iloc[:, 1:-1]]
    columns_entropy = sorted(columns_entropy, key=lambda f: f[1], reverse=True)
    return columns_entropy[0]


def drop_exist_feature(data, best_feature):
    attr = pd.unique(data[best_feature])
    new_data = [(nd, data[data[best_feature] == nd]) for nd in attr]
    new_data = [(n[0], n[1].drop([best_feature], axis=1)) for n in new_data]
    return new_data


def create_tree(data_set, column_count):
    #print(data_set)
    label_list = data_set.iloc[:, -1]
    #label只有一种
    if len(pd.unique(label_list)) == 1:
        return label_list.values[0]
    #训练集一样，选择最多的结果
    if all([len(pd.unique(data_set[col])) == 1 for col in data_set.iloc[:, :-1].columns]):
        return get_most_label(label_list)
    #best_attr: 纹理
    best_attr = get_max_gain(data_set)[0]
    #print(best_attr)
    tree = {best_attr: {}}
    exist_attr = pd.unique(data_set[best_attr])
    if len(exist_attr) != len(column_count[best_attr]):
        no_exist_attr = set(column_count[best_attr]) - set(exist_attr)
        for it in no_exist_attr:
            tree[best_attr][it] = get_most_label(best_attr)
    for item in drop_exist_feature(data_set, best_attr):
        tree[best_attr][item[0]] = create_tree(item[1], column_count)
    return tree


# 获取树的叶子节点数目
def get_num_leafs(decision_tree):
    num_leafs = 0
    first_str = next(iter(decision_tree))
    second_dict = decision_tree[first_str]
    for k in second_dict.keys():
        if isinstance(second_dict[k], dict):
            num_leafs += get_num_leafs(second_dict[k])
        else:
            num_leafs += 1
    return num_leafs


# 获取树的深度
def get_tree_depth(decision_tree):
    max_depth = 0
    first_str = next(iter(decision_tree))
    second_dict = decision_tree[first_str]
    for k in second_dict.keys():
        if isinstance(second_dict[k], dict):
            this_depth = 1 + get_tree_depth(second_dict[k])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


# 绘制节点
def plot_node(node_txt, center_pt, parent_pt, node_type):
    arrow_args = dict(arrowstyle='<-')
    font = FontProperties(fname=r'C:\Windows\Fonts\STXINGKA.TTF', size=15)
    create_plot.ax1.annotate(node_txt,
                             xy=parent_pt,
                             xycoords='axes fraction',
                             xytext=center_pt,
                             textcoords='axes fraction',
                             va="center",
                             ha="center",
                             bbox=node_type,
                             arrowprops=arrow_args,
                             FontProperties=font)


# 标注划分属性
def plot_mid_text(cntr_pt, parent_pt, txt_str):
    font = FontProperties(fname=r'C:\Windows\Fonts\MSYH.TTC', size=10)
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_str, va="center", ha="center", color='red', FontProperties=font)


# 绘制决策树
def plot_tree(decision_tree, parent_pt, node_txt):
    d_node = dict(boxstyle="sawtooth", fc="0.8")
    leaf_node = dict(boxstyle="round4", fc='0.8')
    num_leafs = get_num_leafs(decision_tree)
    first_str = next(iter(decision_tree))
    cntr_pt = (plot_tree.xoff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.yoff)
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, d_node)
    second_dict = decision_tree[first_str]
    plot_tree.yoff = plot_tree.yoff - 1.0 / plot_tree.totalD
    for k in second_dict.keys():
        if isinstance(second_dict[k], dict):
            plot_tree(second_dict[k], cntr_pt, k)
        else:
            plot_tree.xoff = plot_tree.xoff + 1.0 / plot_tree.totalW
            plot_node(second_dict[k], (plot_tree.xoff, plot_tree.yoff), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.xoff, plot_tree.yoff), cntr_pt, k)
    plot_tree.yoff = plot_tree.yoff + 1.0 / plot_tree.totalD


def create_plot(dtree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_num_leafs(dtree))
    plot_tree.totalD = float(get_tree_depth(dtree))
    plot_tree.xoff = -0.5 / plot_tree.totalW
    plot_tree.yoff = 1.0
    plot_tree(dtree, (0.5, 1.0), '')
    plt.show()


if __name__ == "__main__":
    #read data from csv file
    workbook = pd.read_csv("./data/watermelon_3.csv", encoding="gbk", header=0)
    workbook.loc[workbook[workbook.columns[-1]] == '是', workbook.columns[-1]] = '好瓜'
    workbook.loc[workbook[workbook.columns[-1]] == '否', workbook.columns[-1]] = '坏瓜'

    workbook = process_continuous_attr(-2, workbook.iloc[:, -2], workbook)
    workbook = process_continuous_attr(-3, workbook.iloc[:, -3], workbook)

    column_count = dict([(ds, list(pd.unique(workbook[ds]))) for ds in workbook.iloc[:, :-1].columns])
    d_tree = create_tree(workbook, column_count)
    create_plot(d_tree)
