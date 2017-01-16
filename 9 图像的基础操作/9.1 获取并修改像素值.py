# coding=utf-8
"""
获取并修改像素值
"""

import cv2
import numpy as np

IMG_PATH = "../images/messi5.jpg"


if __name__ == '__main__':
    img = cv2.imread(IMG_PATH)

    px = img[100, 100]
    print(px)

    blue = img[100, 100, 0]
    print(blue)

    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()