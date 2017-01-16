#coding=utf-8
"""
形态学转换
目标
• 学习不同的形态学操作，例如腐蚀，膨胀，开运算，闭运算等
• 我们要学习的函数有：cv2.erode()， cv2.dilate()， cv2.morphologyEx()
等
"""



import cv2
import numpy as np
from matplotlib import pyplot as plt


PATH = '../img/j.jpg'


img = cv2.imread(PATH, 0)

# 腐蚀
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)

# 膨胀
dilation = cv2.dilate(img, kernel, iterations=1)

# 开运算
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# 闭运算
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# 形态学梯度
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# 礼帽
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# 黑帽
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)


cv2.imshow('原图', img)
cv2.imshow('腐蚀', erosion)
# cv2.imshow('膨胀', dilation)
# cv2.imshow('开运算', opening)
# cv2.imshow('闭运算', closing)
# cv2.imshow('形态学梯度', gradient)
# cv2.imshow('礼帽', tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.subplot(101), plt.imshow(erosion, 'gray'), plt.title('腐蚀')
# plt.subplot(102), plt.imshow(dilation, 'gray'), plt.title('膨胀')
# plt.show()