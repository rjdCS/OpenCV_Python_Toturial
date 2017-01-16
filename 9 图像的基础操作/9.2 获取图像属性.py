# coding=utf-8
"""
获取图像属性
"""

import cv2
import numpy as np

PATH = "../images/messi5.jpg"

if __name__ == '__main__':
    img = cv2.imread(PATH)

    # img.shape 可以获取图像的形状。他的返回值是一个包含行数、列数、通道数的元组
    print('img.shape:', img.shape)

    # img.size 可以返回图像的像素数目
    print('img.size:', img.size)

    # img.dtype 返回的是图像的数据类型
    print('img.dtype:', img.dtype)