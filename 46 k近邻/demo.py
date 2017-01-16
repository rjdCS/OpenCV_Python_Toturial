# coding=utf-8
# Python Version:3.5.1

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)
# Labels each one either Red or Blue with numbers 0 and 1
responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)
# Take Red families and plot them
red = trainData[responses.ravel()==0]       # 红色是0
plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')
# Take Blue families and plot them
blue = trainData[responses.ravel()==1]      # 蓝色是1
plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')
# plt.show()


newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)
plt.scatter(newcomer[:,0], newcomer[:, 1], 80, 'g', 'o')

# knn = cv2.KNearest()
knn = cv2.ml.KNearest_create()      # for Open-Python 3.0
# 训练算法
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
# 查找结果
# ret, results, neighbours ,dist = knn.find_nearest(newcomer, 3)

# knn.findNearest return:
# results: tor with results of prediction (regression or classification) for each input sample.
# neighborResponses: Optional output values for corresponding neighbors.
# results: Vector with results of prediction (regression or classification) for each input sample.
# dist: Optional output distances from the input vectors to the corresponding neighbors.
ret, results, neighbours, dist = knn.findNearest(newcomer, 3)   # for Open-Python 3.0

print("result: ", results, "\n")
print("neighbours: ", neighbours, "\n")
print("distance: ", dist)
plt.show()