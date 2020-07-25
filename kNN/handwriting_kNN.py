import kNN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from os import listdir


def img2vector(filename):
    returnVect = np.zeros([1, 1024])
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros([m, 1024])
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/trainingDigits/' + fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/' + fileNameStr)
        classifierResult = kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with: ' + str(classifierResult) + ', the real answer is: ' + str(classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print('\nthe total number of error is: ' + str(errorCount) + '\nthe total error rate is: ' + str(errorCount / float(mTest)))


if __name__ == "__main__":
    # img2vector('digits/testDigits/0_13.txt')
    handwritingClassTest()
