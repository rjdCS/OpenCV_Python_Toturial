# coding=utf-8
"""
 图像 ROI （Region Of Interest）

"""

import cv2
import numpy as np

IMG_PATH = "../images/messi5.jpg"

if __name__ == '__main__':
    img = cv2.imread(IMG_PATH)

    # [y, x]
    ball = img[280:340, 330:390]
    print('ball:', ball)
    img[273:333, 100:160] = ball

    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()