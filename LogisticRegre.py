# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import tree
# import cv2
# from sklearn.model_selection import train_test_split  # dung de tach bo test ra
# from sklearn.datasets import load_iris
# # ham train_test_split se tach bo train voi bo test ra rieng biet de xay dung model voi nhung thu vien dataset
# # ham train_test_split se tra ve 4 doi so,  doi so 1 danh cho x_train dung de trainning voi doi so thu 3 y_train
# # 2 doi so con lai tuong tu nhung dung de test
# # Hàm sigmoid


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


# # Load data từ file csv
# data = pd.read_csv('D:\MaDoc\data_classification.csv', header=None).values
# N, d = data.shape
# x = data[:, 0:d - 1].reshape(-1, d - 1)
# y = data[:, 2].reshape(-1, 1)
# true_x = []
# true_y = []
# false_x = []
# false_y = []
# for i in data:
#     if i[2] == 1:
#         true_x.append(i[0])
#         true_y.append(i[1])
#     else:
#         false_x.append(i[0])
#         false_y.append(i[1])


# def predict(features, weights):
#     z = np.dot(features, weights)
#     return sigmoid(z)


# def loss_function(features, labels, weights):
#     n = len(labels)
#     prediction = predict(features, weights)
#     """
#     prediction
#     example : [0.6,0.7,0.5,0.4]
#     """
#     loss_class1 = -labels*np.log(prediction)
#     loss_class2 = -(1-labels)*np.log(1-prediction)
#     loss = loss_class1+loss_class2
#     return np.sum(loss)


# def decisionBoundary(p):
#     if p >= 0.5:
#         return 1
#     else:
#         return 0


# def update_weight(features, labels, weights, learning_rate):
#     n = len(labels)
#     prediction = predict(features, weights)
#     weights_temp = np.dot(features.T, (prediction-labels))/n
#     updated_weight = weights-weights_temp*learning_rate
#     return updated_weight


# def train(features, labels, weights, learning_rate, iter):
#     history_loss = []
#     for i in range(iter):
#         weights = update_weight(features, labels, weights, learning_rate)
#         loss = loss_function(features, labels, weights)
#         history_loss.append(loss)
#     return weights, history_loss


# # # Thêm cột 1 vào dữ liệu x
# x = np.hstack((np.ones((N, 1)), x))
# w = np.array([0., 0.1, 0.1]).reshape(-1, 1)
# # Số lần lặp bước 2
# numOfIteration = 100000
# cost = np.zeros((numOfIteration, 1))
# learning_rate = 0.01
# w, loss = train(x, y, w, learning_rate, numOfIteration)
# x_test = [[2, 9, 1]]
# temp = predict(x_test, w)
# print(temp)
# # print("Value will be predicted for student who slept {} hours and studied {} hours".format(
# #     x_test[1], x_test[2]))
# # if (decisionBoundary(temp) == 1):
# #     print("Predict value is {}. So this student will pass!!".format(
# #         decisionBoundary(temp)))
# # else:
# #     print("Predict value is {}. So this student will fail!!".format(
# #         decisionBoundary(temp)))
# # plt.scatter(true_x, true_y, marker="o", c="y",
# #             edgecolors='none', s=30, label='Pass')
# # plt.scatter(false_x, false_y, marker="o", c="r",
# #             edgecolors='none', s=30, label='Fail')
# # plt.legend(loc=1)
# # plt.xlabel('Studied')
# # plt.ylabel('Slept')
# # plt.show()
# # yTime_series = np.array([i for i in range(numOfIteration)])
# # plt.plot(yTime_series, loss)
# # plt.xlabel("Time")
# # plt.ylabel("Loss")
# # plt.show()

# ----------------------
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# data = pd.read_csv('D:\MaDoc\data_classification.csv', header=None)
# data = data.values

# N, d = data.shape

# features = data[:, :d-1].reshape(-1, d-1)
# labels = data[:, 2].reshape(-1, 1)
# matrix = np.ones((100, 3))
# matrix[:, 1: None] = features
# features = matrix


# def sigmoid(z):
#     return 1/(1 + np.exp(-z))


# def predict(features, weights):
#     z = np.dot(features, weights)

#     return sigmoid(z)


# def decisionBoundary(p):
#     if p >= 0.5:
#         return 1
#     else:
#         return 0


# def loss_function(features, labels, weights):
#     n = len(labels)
#     prediction = predict(features, weights)
#     loss = -labels*np.log(prediction) - (1-labels)*np.log(1-prediction)
#     return np.sum(loss)


# def update_weight(features, labels, weights, learning_rate):
#     n = len(labels)
#     prediction = predict(features, weights)
#     weights_temp = np.dot(features.T, (prediction - labels))/n
#     updated_weight = weights - weights_temp*learning_rate
#     return updated_weight


# def train(features, labels, weights, learning_rate, iter):
#     his = []
#     for i in range(iter):
#         weights = update_weight(features, labels, weights, learning_rate)
#         loss = loss_function(features, labels, weights)
#         his.append(loss)
#     return weights, his


# w = np.array([0., 0.1, 0.1]).reshape(-1, 1)
# w, his = train(features, labels, w, 0.01, 100000)
# # x_test = [1,5,9]
# x_test = [[1, 5, 9]]
# temp = predict(x_test, w)
# print(temp)

# -----

import numpy as np
import pandas as pd

data = pd.read_csv('D:\MaDoc\data_classification.csv', header=None)
data = data.values
N, d = data.shape
features = data[:, :d-1].reshape(-1, d-1)
labels = data[:, d-1].reshape(-1, 1)
matrix = np.ones((100, 3))
matrix[:, 1: 3] = features
features = matrix


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(features, weights):
    z = np.dot(features, weights)
    return sigmoid(z)


def loss_function(features, labels, weights):
    n = len(labels)
    prediction = predict(features, weights)
    loss = - (labels*np.log(prediction) + (1 - labels)*np.log(1-prediction))
    return np.sum(loss)


def update_weights(features, labels, weights, learning_rate):
    n = len(labels)
    prediction = predict(features, weights)
    weights_temp = np.dot(features.T, (prediction-labels))/n
    weights_temp = weights_temp*learning_rate
    weights = weights - weights_temp
    return weights


def train(features, labels, weights, learning_rate, iter):
    his_loss = []
    for i in range(iter):
        weights = update_weights(features, labels, weights, learning_rate)
        loss = loss_function(features, labels, weights)
        his_loss.append(loss)
    return weights, his_loss


w = np.array([0., 0.1, 0.1]).reshape(-1, 1)

weights, his_loss = train(features, labels, w, 0.01, 100000)
x_test = np.array([2, 9, 1]).reshape(-1, 3)

print(predict(x_test, weights))
