# coding=utf-8
"""
目标
• 学习使用 OpenCV 中的函数 cv2.kmeans() 对数据进行分类

输入参数
1. samples: 应该是 np.float32 类型的数据，每个特征应该放在一列。
2. nclusters(K): 聚类的最终数目。
3. criteria: 终止迭代的条件。当条件满足时，算法的迭代终止。它应该是
    一个含有 3 个成员的元组，它们是（typw， max_iter， epsilon）：
    • type 终止的类型：有如下三种选择：
        – cv2.TERM_CRITERIA_EPS 只有精确度 epsilon 满足是
        停止迭代。
        – cv2.TERM_CRITERIA_MAX_ITER 当迭代次数超过阈值
        时停止迭代。
        – cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
        上面的任何一个条件满足时停止迭代。
    • max_iter 表示最大迭代次数。
    • epsilon 精确度阈值
4. attempts: 使用不同的起始标记来执行算法的次数。算法会返回紧密度最好的标记。紧密度也会作为输出被返回。
5. flags：用来设置如何选择起始重心。通常我们有两个选择：cv2.KMEANS_PP_CENTERS
和 cv2.KMEANS_RANDOM_CENTERS。

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


x = np.random.randint(25, 100, 25)  # 生成的随机数n: 25 <= n <= 100
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

