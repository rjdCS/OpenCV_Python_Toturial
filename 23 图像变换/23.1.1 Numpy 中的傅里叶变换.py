"""
Numpy 中的傅里叶变换
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

IMG_PATH = "../images/messi5.jpg"

img = cv2.imread(IMG_PATH, 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# 这里构建振幅图的公式没学过
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()



#使用一个 60x60 的矩形窗口对图像进行掩模操作从而去除低频分量
rows, cols = img.shape
crow, ccol = rows/2 , cols/2
fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

# 使用函数 np.fft.ifftshift() 进行逆平移操作
f_ishift = np.fft.ifftshift(fshift)

# 使用函数 np.ifft2() 进行 FFT 逆变换
img_back = np.fft.ifft2(f_ishift)

# 取绝对值
img_back = np.abs(img_back)

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back)
plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
plt.show()