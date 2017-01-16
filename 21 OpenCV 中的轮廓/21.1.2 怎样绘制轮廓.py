import numpy as np
import cv2

# PATH = '../images/apple.jpg'
PATH = '../img/test3.jpg'

img = cv2.imread(PATH)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)    # cv2.THRESH_BINARY=0
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# img = cv2.drawContour(img, contours, -1, (0,255,0), 3)

# 第一个参数是原始图像。
# 第二个参数是轮廓，一个 Python 列表。
# 第三个参数是轮廓的索引（在绘制独立轮廓是很有用，当设置为 -1 时绘制所有轮廓）。
# 接下来的参数是轮廓的颜色和厚度等。
img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

print('contours:', contours)
array = contours[0]
print('array:', array)

cv2.imshow(PATH, img)
cv2.waitKey(0)
cv2.destroyAllWindows()