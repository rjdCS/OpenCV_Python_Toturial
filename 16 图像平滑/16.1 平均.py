# coding=utf-8
"""
这是由一个归一化卷积框完成的。他只是用卷积框覆盖区域所有像素的平均值来代替中心元素。
可以使用函数 cv2.blur() 和 cv2.boxFilter() 来完这个任务。
可以同看查看文档了解更多卷积框的细节。
我们需要设定卷积框的宽和高。
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

PATH = '../images/opencv-logo.png'

img = cv2.imread(PATH)

blur = cv2.blur(img, (5, 5))

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()