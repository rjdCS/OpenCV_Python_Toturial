# coding=utf-8
"""
图像阈值

目标
• 本节你将学到简单阈值，自适应阈值， Otsu’s 二值化等
• 将要学习的函数有 cv2.threshold， cv2.adaptiveThreshold 等。
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

PATH = '../images/sudoku.png'

img = cv2.imread(PATH, 0)

# 中值滤波
img = cv2.medianBlur(img, 5)

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 11 为 Block size, 2 为 C 值
"""
• Adaptive Method- 指定计算阈值的方法。
– cv2.ADPTIVE_THRESH_MEAN_C：阈值取自相邻区域的平
均值
– cv2.ADPTIVE_THRESH_GAUSSIAN_C：阈值取值相邻区域
的加权和，权重为一个高斯窗口。
• Block Size - 邻域大小（用来计算阈值的区域大小）。
• C - 这就是是一个常数，阈值就等于的平均值或者加权平均值减去这个常
数
"""
th2 = cv2.adaptiveThreshold(img, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            thresholdType=cv2.THRESH_BINARY,blockSize=11,C=2)
th3 = cv2.adaptiveThreshold(img, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()