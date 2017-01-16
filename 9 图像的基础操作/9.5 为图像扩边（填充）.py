#coding=utf-8

"""
如果你想在图像周围创建一个边，就像相框一样，你可以使用 cv2.copyMakeBorder()
函数。这经常在卷积运算或 0 填充时被用到。这个函数包括如下参数：
• src 输入图像
• top, bottom, left, right 对应边界的像素数目。
• borderType 要添加那种类型的边界，类型如下
– cv2.BORDER_CONSTANT 添加有颜色的常数值边界，还需要
下一个参数（value）。
– cv2.BORDER_REFLECT 边界元素的镜像。比如: fedcba|abcdefgh|hgfedcb
– cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT
跟上面一样，但稍作改动。例如: gfedcb|abcdefgh|gfedcba
– cv2.BORDER_REPLICATE 重复最后一个元素。例如: aaaaaa|
abcdefgh|hhhhhhh
– cv2.BORDER_WRAP 不知道怎么说了, 就像这样: cdefgh|
abcdefgh|abcdefg
• value 边界颜色，如果边界的类型是 cv2.BORDER_CONSTANT

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

PATH = "../images/opencv-logo-white.png"

BLUE = [255, 0, 0]

img1 = cv2.imread(PATH)

replicate = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT,  value=BLUE)

plt.subplot(231), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')

plt.show()