import numpy as np


def loadTrainSet():
    fr = open('mnist_dataset/mnist_train.csv', 'r')
    trainSet = [line.split(',') for line in fr.readlines()]
    trainSet = np.mat(trainSet, dtype='float32')
    trainInput = trainSet[:, 1:]
    trainLabelList = trainSet[:, 0]
    fr.close()
    return trainInput, trainLabelList


def loadTestSet():
    fr = open('mnist_dataset/mnist_test.csv', 'r')
    testSet = [line.split(',') for line in fr.readlines()]
    testSet = np.mat(testSet)
    testInput = testSet[:, 1:]
    testLabelList = testSet[:, 0]
    fr.close()
    return testInput, testLabelList


class threeLayersNeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.lr = learningrate

    # input: np.mat(m, 1) targets: np.mat(10, 1)
    def train(self, inputs, targets):
        hidden_inputs = self.wih * inputs
        hidden_outputs = self.sigmoid(hidden_inputs)
        final_inputs = self.who * hidden_outputs
        final_outputs = self.sigmoid(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = self.who.T * output_errors
        self.who += self.lr * np.multiply(np.multiply(output_errors, final_outputs),\
             1 - final_outputs) * hidden_outputs.T
        self.wih += self.lr * np.multiply(np.multiply(hidden_errors, hidden_outputs),\
             1 - hidden_outputs) * inputs.T

    # input: np.mat(m, 1)
    def query(self, inputs):
        hidden_inputs = self.wih * inputs
        hidden_outputs = self.sigmoid(hidden_inputs)
        final_inputs = self.who * hidden_outputs
        final_outputs = self.sigmoid(final_inputs)
        return final_outputs

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))


if __name__ == "__main__":
    trainInput, trainLabelList = loadTrainSet()
    nn = threeLayersNeuralNetwork(784, 200, 10, 0.3)
    for i in range(len(trainLabelList)):
        scaledTrainInput = np.mat(np.asfarray(trainInput[i, :]) / 255.0 * 0.99 + 0.01).T
        targetNum = int(trainLabelList[i])
        targets = np.mat(np.ones((10, 1))) * 0.01
        targets[targetNum, 0] = 0.99
        nn.train(scaledTrainInput, targets)

    testInput, testLabelList = loadTestSet()
    scoreCard = []
    for i in range(len(testLabelList)):
        scaledTestInput = np.mat(np.asfarray(testInput[i, :]) / 255.0 * 0.99 + 0.01).T
        targetNum = int(testLabelList[i])
        predictNum = int(np.argmax(nn.query(scaledTestInput), 0))
        if targetNum == predictNum:
            scoreCard.append(1)
        else:
            scoreCard.append(0)
    print(sum(scoreCard) / len(testLabelList))
