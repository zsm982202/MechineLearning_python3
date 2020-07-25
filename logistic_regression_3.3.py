"""
Author: zsm
Created on: 2020.5.24 23:00
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def newton(X, y):
    """
    Input:
        X: np.array with shape [m, d + 1]. Input.
        y: np.array with shape [m, 1]. Label.
    Return:
        beta: np.array with shape [1, d + 1]. Optimal params with newton method
    """
    m = X.shape[0]
    d = X.shape[1] - 1
    #initialization
    beta = np.ones((1, d + 1)) * 0.1

    while True:
        #z: np.array with shape[m, 1]
        z = np.dot(X, beta.T)
        #p1: np.array with shape[m, 1]
        p1 = np.exp(z) / (1 + np.exp(z))
        #first_order: np.array with shape[1, d + 1]
        first_order = np.sum(X * (p1 - y), 0, keepdims=True)
        #second_order: np.array with shape[1, 1]
        #0表示按列求和,keepdims保持其二维特性
        second_order = np.sum(X.dot(X.T).dot(p1 * (1 - p1)), 0, keepdims=True)

        [rows, cols] = first_order.shape
        #flag == True: break
        flag = False
        delta = np.linalg.inv(second_order) * first_order
        for item in delta:
            for it in item:
                if (it > 1e-5):
                    flag = True
        if flag == False:
            break
        beta -= delta

    l = np.sum(-y * z + np.log(1 + np.exp(z)))
    print("newton:" + str(l))
    return beta


def gradDescent(X, y):
    """
    Input:
        X: np.array with shape [m, d + 1]. Input.
        y: np.array with shape [m, 1]. Label.
    Return:
        beta: np.array with shape [1, d + 1]. Optimal params with grad descent method
    """
    m = X.shape[0]
    d = X.shape[1] - 1
    lr = 0.05

    #initialization
    beta = np.ones((1, d + 1)) * 0.1

    while True:
        #z: np.array with shape[m, 1]
        z = np.dot(X, beta.T)
        #p: np.array with shape[m, d + 1]
        p = X * (np.exp(z) / (1 + np.exp(z)) - y)
        #first_order: np.array with shape[1, d + 1]
        first_order = np.sum(p, 0, keepdims=True)  #0表示按列求和,keepdims保持其二维特性
        [rows, cols] = first_order.shape
        #flag == True: break
        flag = False
        for item in first_order:
            for it in item:
                if (it > 1e-5):
                    flag = True
        if flag == False:
            break
        beta -= lr * first_order

    l = np.sum(-y * z + np.log(1 + np.exp(z)))
    print("grad descent:" + str(l))
    return beta


if __name__ == "__main__":
    #read data from csv file
    workbook = pd.read_csv("./data/watermelon_3a.csv", header=None)
    #this is the extension of x
    workbook.insert(3, "3", 1)
    X = workbook.values[:, 1:4]
    y = workbook.values[:, -1].reshape(-1, 1)  #-1表示n

    #plot training data
    positive_data = workbook.values[workbook.values[:, 4] == 1.0, :]
    negative_data = workbook.values[workbook.values[:, 4] == 0, :]
    plt.plot(positive_data[:, 1], positive_data[:, 2], 'bo')
    plt.plot(negative_data[:, 1], negative_data[:, 2], 'r+')

    #get optimal params beta with newton method
    beta = newton(X, y)
    newton_left = -(beta[0, 0] * 0.1 + beta[0, 2]) / beta[0, 1]
    newton_right = -(beta[0, 0] * 0.9 + beta[0, 2]) / beta[0, 1]
    plt.plot([0.1, 0.9], [newton_left, newton_right], 'g-')

    #get optimal params beta with gradient descent method
    beta = gradDescent(X, y)
    grad_descent_left = -(beta[0, 0] * 0.1 + beta[0, 2]) / beta[0, 1]
    grad_descent_right = -(beta[0, 0] * 0.9 + beta[0, 2]) / beta[0, 1]
    plt.plot([0.1, 0.9], [grad_descent_left, grad_descent_right], 'y-')

    plt.xlabel('density')
    plt.ylabel('sugar rate')
    plt.title("LR")
    plt.show()
