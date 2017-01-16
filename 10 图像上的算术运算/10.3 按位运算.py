# coding=utf-8
"""
按位运算
"""

import cv2
import numpy as np

IMG_PATH1 = '../images/messi5.jpg'
IMG_PATH2 = '../images/opencv-logo-white.png'


# 加载图像
img1 = cv2.imread(IMG_PATH1)
img2 = cv2.imread(IMG_PATH2)

# I want to put logo on top-left corner, So I create a ROI
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
# 取 roi 中与 mask 中不为零的值对应的像素的值，其他值为 0
# 注意这里必须有 mask=mask 或者 mask=mask_inv, 其中的 mask= 不能忽略
img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

# 取 roi 中与 mask_inv 中不为零的值对应的像素的值，其他值为 0。
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols ] = dst
cv2.imshow('res', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()