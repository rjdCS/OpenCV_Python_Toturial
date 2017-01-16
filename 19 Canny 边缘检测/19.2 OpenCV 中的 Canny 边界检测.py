# coding=utf-8
"""
Canny边缘检测
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

PATH = '../images/messi5.jpg'

img = cv2.imread(PATH, 0)

# 第一个参数是输入图像。
# 第二和第三个分别是 minVal 和 maxVal。
# 第三个参数设置用来计算图像梯度的 Sobel卷积核的大小，默认值为 3。
# 最后一个参数是 L2gradient，它可以用来设定求梯度大小的方程。
# 如果设为 True，就会使用我们上面提到过的方程，否则使用方程： Edge−Gradient(G) = jG2 xj + jG2 yj 代替，默认值为 False。
edges = cv2.Canny(img, 100, 200)


plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])

plt.show()