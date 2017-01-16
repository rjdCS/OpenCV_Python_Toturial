"""
目标
本节我们将要学习：
• 另外一个角点检测技术： Shi-Tomasi 焦点检测
• 函数： cv2.goodFeatureToTrack()

"""


import numpy as np
import cv2
from matplotlib import pyplot as plt

PATH = '../images/blox.jpg'

img = cv2.imread(PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 通常情况下，输入的应该是灰度图像。
# 然后确定你想要检测到的角点数目。
# 再设置角点的质量水平， 0 到 1 之间。它代表了角点的最低质量，低于这个数的所有角点都会被忽略。
# 最后在设置两个角点之间的最短欧式距离。
corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
# 返回的结果是 [[ 311.,  250.]] 两层括号的数组。

corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)
plt.imshow(img), plt.show()