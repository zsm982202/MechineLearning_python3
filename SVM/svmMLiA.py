import random
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        if float(lineArr[2]) == 1:
            labelMat.append(1.0)
        else:
            labelMat.append(-1.0)
    return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while (i == j):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def smoSimple(dataMat, classLabels, C, toler, maxIter):
    '''
    @dataMat    ：数据列表
    @classLabels：标签列表
    @C          ：权衡因子（增加松弛因子而在目标优化函数中引入了惩罚项）
    @toler      ：容错率
    @maxIter    ：最大迭代次数
    '''
    #将列表形式转为矩阵或向量形式
    dataMatrix = np.mat(dataMat)
    labelMat = np.mat(classLabels).transpose()
    #初始化b=0，获取矩阵行列
    b = 0
    m, n = np.shape(dataMatrix)
    #新建一个m行1列的向量
    alphas = np.mat(np.zeros((m, 1)))
    #迭代次数为0
    iters = 0
    while (iters < maxIter):
        #改变的alpha对数
        alphaPairsChanged = 0
        #遍历样本集中样本
        for i in range(m):
            #计算支持向量机算法的预测值
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            #计算预测值与实际值的误差
            Ei = fXi - float(labelMat[i])
            #如果不满足KKT条件，即labelMat[i]*fXi<1(labelMat[i]*fXi-1<-toler)
            #and alpha<C 或者labelMat[i]*fXi>1(labelMat[i]*fXi-1>toler)and alpha>0
            if (((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0))):
                #随机选择第二个变量alphaj
                j = selectJrand(i, m)
                #计算第二个变量对应数据的预测值
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                #计算与测试与实际值的差值
                Ej = fXj - float(labelMat[j])
                #记录alphai和alphaj的原始值，便于后续的比较
                alphaIold = np.copy(alphas[i])
                alphaJold = np.copy(alphas[j])
                #如何两个alpha对应样本的标签不相同
                if (labelMat[i] != labelMat[j]):
                    #求出相应的上下边界
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue
                #根据公式计算未经剪辑的alphaj
                #------------------------------------------
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T\
                     - dataMatrix[j, :] * dataMatrix[j, :].T
                #如果eta>=0,跳出本次循环
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                #------------------------------------------
                #如果改变后的alphaj值变化不大，跳出本次循环
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    # print("j not moving enough")
                    continue
                #否则，计算相应的alphai值
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                #再分别计算两个alpha情况下对于的b值
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T\
                     - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T\
                     - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                #如果0<alphai<C,那么b=b1
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                    #否则如果0<alphai<C,那么b=b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                    #否则，alphai，alphaj=0或C
                else:
                    b = (b1 + b2) / 2.0
                #如果走到此步，表明改变了一对alpha值
                alphaPairsChanged += 1
                print("iters: %d i:%d,paird changed %d" % (iters, i, alphaPairsChanged))
        #最后判断是否有改变的alpha对，没有就进行下一次迭代
        if (alphaPairsChanged == 0):
            iters += 1
            #否则，迭代次数置0，继续循环
        else:
            iters = 0
        print("iteration number: %d" % iters)
    #返回最后的b值和alpha向量
    return b, alphas


def plot_point(dataMat, labelMat, alphas, b):
    data1 = np.mat(dataMat)[np.array(labelMat) == 1, :]
    data2 = np.mat(dataMat)[np.array(labelMat) == -1, :]
    plt.plot(data1[:, 0], data1[:, 1], 'bo')
    plt.plot(data2[:, 0], data2[:, 1], 'r+')

    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)  #转为数组求w
    omega = 0  #初始化w
    for i in range(np.shape(dataMat)[0]):
        omega += alphas[i] * labelMat[i] * dataMat[i].T  #通过求导后得到的w公式，求和得到w
    # print(sum)

    for i, alpha in enumerate(alphas):  #根据kkt条件，选取alpha不为0的点作为支持向量
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter(x, y, linewidths=1.5, edgecolor='green')

    x1 = np.arange(-4, 4, 0.01)
    y1 = (omega[0] * x1 + float(b)) / (-1 * omega[1])  #通过原始公式推导出超平面
    plt.plot(x1, y1)
    plt.show()


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def kernelTrans(X, A, kTup):
    '''
    RBF kernel function
    '''
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1]**2))

    else:
        raise NameError('huston ---')
    return K


def calcEk(oS, k):
    '''计算预测误差'''
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


#修改选择第二个变量alphaj的方法
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    #将误差矩阵每一行第一列置1，以此确定出误差不为0
    #的样本
    oS.eCache[i] = [1, Ei]
    #获取缓存中Ei不为0的样本对应的alpha列表
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    #在误差不为0的列表中找出使abs(Ei-Ej)最大的alphaj
    if (len(validEcacheList) > 0):
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        #否则，就从样本集中随机选取alphaj
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


#更新误差矩阵
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  #changed for kernel
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  #update i by the same amount as j
        updateEk(oS, i)  #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):  #full Platt SMO
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  #go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False  #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))  #C=200 important
    # plot_point(dataArr, labelArr, alphas, b)
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]  #get matrix of only support vectors
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)  #load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 50)):
    dataArr, labelArr = loadImages('digits/trainingDigits')
    print(111)
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m, n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadImages('digits/testDigits')
    errorCount = 0
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + float(b)
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


if __name__ == "__main__":
    # dataArr, labelArr = loadDataSet('testSet.txt')
    # b, alphas = smoP(dataArr, labelArr, 0.6, 0.0001, 40)
    # plot_point(dataArr, labelArr, alphas, b)

    # testRbf()


    # testDigits()
