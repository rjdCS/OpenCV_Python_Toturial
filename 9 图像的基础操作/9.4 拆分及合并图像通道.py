# coding=utf-8
"""
拆分及合并图像通道
"""

import cv2
import numpy as np

IMG_PATH = "../images/messi5.jpg"

if __name__ == '__main__':
    img = cv2.imread(IMG_PATH)

    # 把 BGR 拆分成单个通道
    b, g, r = cv2.split(img)
    # 把独立通道的图片合并成一个 BGR 图像
    # img = cv2.merge(b, g, r)

    b = img[:, :, 0]


    # 使所有像素的红色通道值都为 0
    print('使所有像素的红色通道值都为 0')
    img = cv2.imread(IMG_PATH)

    img[:, :, 2] = 0

    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()