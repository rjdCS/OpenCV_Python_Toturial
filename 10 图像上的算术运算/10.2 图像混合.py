# coding=utf-8
"""
图像混合
"""

import cv2
import numpy as np

IMG_PATH1 = '../images/ml.png'
IMG_PATH2 = '../images/opencv-logo-white.png'
# IMG_PATH2 = '../images/ml.png'

if __name__ == '__main__':

    img1 = cv2.imread(IMG_PATH1)
    img2 = cv2.imread(IMG_PATH2)

    dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)

    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindow()