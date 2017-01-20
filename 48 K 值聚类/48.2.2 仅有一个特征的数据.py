# coding=utf-8
"""
目标
• 学习使用 OpenCV 中的函数 cv2.kmeans() 对数据进行分类
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

x = np.random.randint(25, 100, 25)
y = np.random.randint(175, 255, 25)
z = np.hstack((x, y))
z = z.reshape((50, 1))
z = np.float32(z)
plt.subplot(221)
plt.hist(z, 256, [0, 256])
# plt.show()

# 设置终止条件：算法执行 10 次迭代或者精确度 epsilon = 1.0。
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# 返回值有紧密度（compactness）, 标志和中心
# Apply KMeans
compactness, labels, centers = cv2.kmeans(z, 2, None, criteria, 10, flags)

A = z[labels == 0]
B = z[labels == 1]

# 将 A 组数用红色表示，将 B 组数据用蓝色表示，重心用黄色表示。
# Now plot 'A' in red, 'B' in blue, 'centers' in yellow
plt.subplot(222)
plt.hist(A, 256, [0, 256], color='r')
plt.hist(B, 256, [0, 256], color='b')
plt.hist(centers, 32, [0, 256], color='y')
plt.show()
