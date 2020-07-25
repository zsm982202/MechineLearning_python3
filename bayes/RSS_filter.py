import bayes
import re
import numpy as np


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict, reverse=True)
    return sortedFreq[:30]


def stopWords():
    import re
    wordList = open('stopword.txt').read()
    listOfTokens = re.split('\n', wordList)
    return [tok.lower() for tok in listOfTokens]


def localWords(feed1, feed0):
    import textparser
    import feedparser
    import pyquery
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = bayes.textPrase(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = bayes.textPrase(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    stopwords = stopWords()

    for pairW in stopwords:
        if pairW in vocabList:
            vocabList.remove(pairW)

    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bayes.bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = bayes.trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bayes.bagOfWords2VecMN(vocabList, docList[docIndex])
        if bayes.classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    errorRate = float(errorCount) / len(testSet)
    print('the error rate is: ', errorRate)
    return vocabList, p0V, p1V, errorRate


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V, errorRate = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -5.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -5.0:
            topNY.append([vocabList[i], p1V[i]])
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
    for item in sortedSF:
        #打印每个二元列表中的单词字符串元素
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**')
    for item in sortedNY:
        print(item[0])


if __name__ == "__main__":
    import feedparser
    ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
    sf = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
    # errorRateList = []
    # for i in range(100):
    #     vocabList, pSF, pNY, errorRate = localWords(ny, sf)
    #     errorRateList.append(errorRate)
    # print(sum(errorRateList) / len(errorRateList))
    getTopWords(ny, sf)
