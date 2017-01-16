"""
读取和显示图片
"""

import cv2
import numpy as np

IMG_PATH = "../images/messi5.jpg"

if __name__ == '__main__':
    img = cv2.imread(IMG_PATH)

    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)              # 如果不添最后一句，在IDLE中执行窗口直接无响应。在命令行中执行的话，则是一闪而过。
    cv2.destroyAllWindows()