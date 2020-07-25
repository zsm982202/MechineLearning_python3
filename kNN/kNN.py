# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties


def createDataSet():
    group = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, [dataSetSize, 1]) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    #[1.48660687 1.41421356 0.         0.1       ]
    distances = sqDistances**0.5
    #从小到大索引[2 3 1 0]
    sortedDistIndicies = np.argsort(distances)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount, reverse=True)
    return sortedClassCount[0]


def label2int(labels):
    label_dict = {}
    label_int = []
    label_num = 1
    for label in labels:
        if label not in label_dict:
            label_dict[label] = label_num
            label_num += 1
    for label in labels:
        label_int.append(label_dict[label])
    return label_int


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    classLabelVector = label2int(classLabelVector)
    return returnMat, classLabelVector


def plot0():
    matrix, labels = file2matrix('datingTestSet.txt')
    labels = np.array(labels)
    font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
    plt.figure(figsize=(8, 5), dpi=80)
    axes = plt.subplot(111)
    # 将三类数据分别取出来
    # x轴代表飞行的里程数
    # y轴代表玩视频游戏的百分比
    t1 = matrix[labels == 3]
    t2 = matrix[labels == 2]
    t3 = matrix[labels == 1]

    type1 = axes.scatter(t1[:, 0], t1[:, 1], s=20, c='red')
    type2 = axes.scatter(t2[:, 0], t2[:, 1], s=40, c='green')
    type3 = axes.scatter(t3[:, 0], t3[:, 1], s=50, c='blue')
    plt.xlabel(u'每年获取的飞行里程数', fontproperties=font_set)
    plt.ylabel(u'玩视频游戏所消耗的事件百分比', fontproperties=font_set)
    axes.legend((type1, type2, type3), (u'不喜欢', u'魅力一般', u'极具魅力'), loc=2, prop=font_set)

    plt.show()


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: ' + str(classifierResult) + ', the real answer is: ' + str(datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print('the total error rate is: ' + str(errorCount / numTestVecs))


if __name__ == "__main__":
    # group, labels = createDataSet()
    # matrix, labels = file2matrix('datingTestSet.txt')
    # normDataSet, ranges, minVals = autoNorm(matrix)

    datingClassTest()
    #print(classify0([0, 0], group, labels, 3))
