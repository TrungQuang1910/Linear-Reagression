import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('D:\MaDoc\Advertising.csv', header=0)
data = data.values
print(data)
X = data[:, 2]
Y = data[:, 4]
# print(X)
# print(Y)
# plt.scatter(X, Y, marker='o', color='r')
# plt.show()


def predict(new_radio, weight, bias):
    return weight*new_radio + bias


def error_function(X, Y, weight, bias):
    n = len(X)
    sum_error = 0
    for i in range(n):
        sum_error += (Y[i] - (weight*X[i] + bias))**2
    return sum_error/n


def update_weight_bias(X, Y, weight, bias, learning_rate):
    n = len(X)
    weight_temp = 0
    bias_temp = 0
    for i in range(n):
        weight_temp += (-2*X[i]*(Y[i] - (weight*X[i] + bias)))
        bias_temp += (-2*(Y[i]-(weight*X[i]+bias)))
    weight = weight - (weight_temp/n)*learning_rate
    bias = bias - (bias_temp/n)*learning_rate
    return weight, bias


def train(X, Y, weight, bias, learning_rate, iter):
    his = []
    for i in range(iter):
        weight, bias = update_weight_bias(X, Y, weight, bias, learning_rate)
        sum_error = error_function(X, Y, weight, bias)
        his.append(sum_error)
    return his, weight, bias


his, weight, bias = train(X, Y, 11, 20, 0.001, 30)
x = [i for i in range(30)]
# plt.scatter(x, his)
plt.plot(x, his)


print(predict(19, weight, bias))
plt.show()


