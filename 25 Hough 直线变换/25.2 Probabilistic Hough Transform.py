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
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
minLineLength = 100
maxLineGap = 10

# cv2.HoughLinesP()
# minLineLength - 线的最短长度。比这个短的线都会被忽略。
# MaxLineGap - 两条线段之间的最大间隔，如果小于此值，这两条直线就被看成是一条直线。
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# cv2.imwrite('houghlines5.jpg',img)
cv2.imshow('sudoku', img)
cv2.waitKey(0)
cv2.destroyAllWindows()