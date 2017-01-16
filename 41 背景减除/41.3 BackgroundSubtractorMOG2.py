
"""
这个也是以高斯混合模型为基础的背景/前景分割算法。它是以 2004 年和 2006 年 Z.Zivkovic 的两篇文章为基础的。
这个算法的一个特点是它为每一个像素选择一个合适数目的高斯分布。
（上一个方法中我们使用是 K 高斯分布）。这样就会对由于亮度等发生变化引起的场景变化产生更好的适应。

和前面一样我们需要创建一个背景对象。但在这里我们我们可以选择是否检测阴影。
如果 detectShadows = T rue（默认值），它就会检测并将影子标记出来，但是这样做会降低处理速度。影子会被标记为灰色。
"""

import numpy as np
import cv2

PATH = '../images/'

cap = cv2.VideoCapture(PATH+'vtest.avi')
# 如果 detectShadows = T rue（默认值），它就会检测并将影子标记出来，但是这样做会降低处理速度。影子会被标记为灰色。
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()