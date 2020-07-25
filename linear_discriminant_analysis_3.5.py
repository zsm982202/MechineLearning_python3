"""
Author: zsm
Created on: 2020.5.25 14:30
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def LDA(X0, X1):
    """
    Get the optimal params of LDA model given training data.
    Input:
        X0: np.array with shape [N1, d]
        X1: np.array with shape [N2, d]
    Return:
        omega: np.array with shape [1, d]. Optimal params of LDA.
    """
    #u0, u1: np.array with shape [1, d]
    u0 = np.mean(x0, 0, keepdims=True)
    u1 = np.mean(x1, 0, keepdims=True)
    Sw = (x0 - u0).T.dot(x0 - u0) + (x1 - u1).T.dot(x1 - u1)
    w = np.linalg.inv(Sw).dot((u0 - u1).T)
    return w


if __name__ == "__main__":
    #read data from csv file
    workbook = pd.read_csv("./data/watermelon_3a.csv", header=None)

    positive_data = workbook.values[workbook.values[:, -1] == 1.0, :]
    negative_data = workbook.values[workbook.values[:, -1] == 0, :]

    x0 = positive_data[:, 1:3]  #column from [1,3)
    x1 = negative_data[:, 1:3]

    #LDA
    w = LDA(x0, x1)

    #plot
    plt.plot(positive_data[:, 1], positive_data[:, 2], 'bo')
    plt.plot(negative_data[:, 1], negative_data[:, 2], 'r+')
    lda_left = 0
    lda_right = -(w[0] * 0.9) / w[1]
    plt.plot([0, 0.9], [lda_left, lda_right], 'g-')


    plt.xlabel('density')
    plt.ylabel('sugar rate')
    plt.title("LDA")
    plt.show()

