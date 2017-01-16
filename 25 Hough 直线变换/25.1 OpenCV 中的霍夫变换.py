# coding=utf-8
"""
OpenCV 中的霍夫变换

目标
• 理解霍夫变换的概念
• 学习如何在一张图片中检测直线
• 学习函数： cv2.HoughLines()， cv2.HoughLinesP()
"""

import cv2
import numpy as np

PATH = '../images/sudoku.png'

img = cv2.imread(PATH)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = cv2.imread(PATH, 0)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# cv2.imwrite('houghlines3.jpg', img)
cv2.imshow('sudoku', img)
cv2.waitKey(0)
cv2.destroyAllWindows()