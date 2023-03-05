import pandas
import os
import math
import random
from sklearn.naive_bayes import GaussianNB
import numpy as np
from diffprivlib import models as dp
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# read a dataset
def read(filename):
    rec = []
    try:
        with open(filename, 'r') as lines:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                temp_line = line.split(',')
                new_line = []
                for x in temp_line:
                    new_line.append(x.strip())
                rec.append(new_line)
        return rec
    except:
        print('cannot open file ' + filename)

# Convert string column to float


def convert_float(dataset):
    column = [0, 1, 2, 3]
    for i in column:
        for line in dataset:
            line[i] = float(line[i].strip())

# Convert string column to integer


def convert_to_int(dataset):
    # dict = {
    #     "Iris-setosa" : 0,
    #     "Iris-versicolor" : 1,
    #     "Iris-virginica": 2
    # }
    for line in dataset:
        if line[4] == "Iris-setosa":
            line[4] = 0
        elif line[4] == "Iris-versicolor":
            line[4] = 1
        elif line[4] == "Iris-virginica":
            line[4] = 2


# read iris data
rec = read('iris.data')
convert_float(rec)
convert_to_int(rec)

# split to test and training data
train_x = []
train_y = []
test_x = []
test_y = []
for i in range(len(rec)):
    if i in range(0, 10):
        test_x.append(rec[i][0:4])
        test_y.append(rec[i][4])
    elif i in range(10, 50):
        train_x.append(rec[i][0:4])
        train_y.append(rec[i][4])
    elif i in range(50, 60):
        test_x.append(rec[i][0:4])
        test_y.append(rec[i][4])
    elif i in range(60, 100):
        train_x.append(rec[i][0:4])
        train_y.append(rec[i][4])
    elif i in range(100, 110):
        test_x.append(rec[i][0:4])
        test_y.append(rec[i][4])
    else:
        train_x.append(rec[i][0:4])
        train_y.append(rec[i][4])


# Calculate accuracy percentage


def accuracy(actual, predicted):
    correct_prediction = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct_prediction += 1
    return float(correct_prediction / len(actual)) * 100.0


gnb = GaussianNB()
y_pred = gnb.fit(train_x, train_y).predict(test_x)
print("Part a")
print("Number of mislabeled points out of a total {} points : {}".format(
      len(test_x), (test_y != y_pred).sum()))
print("Non-private test accuracy : {}".format(accuracy(test_y, y_pred)))

# Part b
print("Part b")
# diff_p = dp.GaussianNB(epsilon=float("0.1"))
diff_p = dp.GaussianNB()
y_pred_dp = diff_p.fit(train_x, train_y).predict(test_x)
print("epsilon: " + str(diff_p.epsilon))
print("Number of mislabeled points out of a total {} points : {}".format(
      len(test_x), (test_y != y_pred_dp).sum()))
print("Differentially private test accuracy : {}".format(
    accuracy(test_y, y_pred_dp)))


max_log = 0
value = [0, 1, 2]
for j in value:
    non_DP_count = 0
    DP_count = 0
    for i in y_pred:
        if i == 0:
            non_DP_count += 1

    for i in y_pred_dp:
        if i == 0:
            DP_count += 1

    Pr_non_DP = non_DP_count / len(y_pred)
    Pr_DP = DP_count / len(y_pred_dp)
    # log_pr = max(abs(math.log(float(Pr_non_DP / Pr_DP))),
    #              abs(math.log(float(Pr_DP / Pr_non_DP))))
    log_pr = abs(math.log(float(Pr_non_DP / Pr_DP)))
    if log_pr > max_log:
        max_log = log_pr
print(str(max_log) + " <= (epsilon = 1)")


e = [0.5, 1, 2, 4, 8, 16]
acc_result = []
print("Part d")
for i in e:
    diff_p = dp.GaussianNB(epsilon=float(i))
    y_pred_dp = diff_p.fit(train_x, train_y).predict(test_x)
    acc_result.append(accuracy(test_y, y_pred_dp))
    print("epsilon: " + str(diff_p.epsilon))
    print("Number of mislabeled points out of a total {} points : {}".format(
        len(test_x), (test_y != y_pred_dp).sum()))
    print("Differentially private test accuracy : {}".format(
        accuracy(test_y, y_pred_dp)))
    print("Precision: " + str(precision_score(test_y, y_pred_dp, average='macro')))
    print("Recall: " + str(recall_score(test_y, y_pred_dp, average='macro')))
plt.plot(e, acc_result)
plt.title("Differentially private Naive Bayes accuracy")
plt.xlabel("epsilon")
plt.ylabel("Accuracy")
plt.show()


# add laplace noise
def laplacian_noise(e):
    return np.random.laplace(e)
