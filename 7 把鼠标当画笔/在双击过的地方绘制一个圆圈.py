# coding=utf-8
"""
在双击过的地方绘制一个圆圈
"""

import cv2
import numpy as np


# mouse callback function
def draw_circle(event, x, y, flags, param):
    """双击画圆"""
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)

if __name__ == '__main__':

    # 创建图像与窗口并将窗口与回调函数绑定
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(20)&0xFF==27:    # 27表示按下Esc键
            break
    cv2.destroyAllWindows()