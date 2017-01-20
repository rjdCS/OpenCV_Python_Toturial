# coding=utf-8
"""
颜色量化就是减少图片中颜色数目的一个过程。
为什么要减少图片中的颜色呢？减少内存消耗！
有些设备的资源有限，只能显示很少的颜色。在这种情况下就需要进行颜色量化。
我们使用 K 值聚类的方法来进行颜色量化。

现在有 3 个特征： R， G， B。
所以我们需要把图片数据变形成 Mx3（M 是图片中像素点的数目）的向量。
聚类完成后，我们用聚类中心值替换与其同组的像素值，这样结果图片就只含有指定数目的颜色了。

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

img = cv2.imread('../images/home.jpg')
Z = img.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))


# cv2.imshow('res2', res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


plt.subplot(221)
plt.imshow(img)
plt.title('img')
print('img:', sys.getsizeof(img), 'bytes')

plt.subplot(222)
plt.imshow(res2)
plt.title('res2,k=%d' % K)
print('res2:', sys.getsizeof(res2), 'bytes')

plt.show()
