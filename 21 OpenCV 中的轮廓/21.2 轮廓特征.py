

import cv2
import numpy as np

PATH = '../img/test3.jpg'

# img = cv2.imread(PATH, 0)
img = cv2.imread(PATH)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 矩
cnt = contours[0]
M = cv2.moments(cnt)
print(M)

# 计算出对象的重心
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print('cx:', cx)
print('cy:', cy)

# 面积
area = cv2.contourArea(cnt)
print('area:', area)

# 轮廓周长
perimeter = cv2.arcLength(cnt, True)
print('perimeter:', perimeter)


# 轮廓近似
epsilon = 0.1*cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
print('approx:', approx)

# 边界矩形
x, y, w, h = cv2.boundingRect(cnt)
print('x:', x, 'y:', y, 'w:', w, 'h:', h)
img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.imshow(PATH, img)
cv2.waitKey(0)
cv2.destroyAllWindows()