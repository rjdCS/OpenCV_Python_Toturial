
"""
使用 GrabCut 算法进行交互式前景提取

目标
在本节中我们将要学习：
• GrabCut 算法原理，使用 GrabCut 算法提取图像的前景
• 创建一个交互是程序完成前景提取

原理
从用户的角度来看它到底是如何工作的呢？开始时用户需要用一个矩形将
前景区域框住（前景区域应该完全被包括在矩形框内部）。然后算法进行迭代式
分割直达达到最好结果。但是有时分割的结果不够理想，比如把前景当成了背
景，或者把背景当成了前景。在这种情况下，就需要用户来进行修改了。用户
只需要在不理想的部位画一笔（点一下鼠标）就可以了。画一笔就等于在告诉
计算机：“嗨，老兄，你把这里弄反了，下次迭代的时候记得改过来呀！”。然后，
在下一轮迭代的时候你就会得到一个更好的结果了。
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

PATH = '../images/messi5.jpg'



img = cv2.imread(PATH)
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (50, 50, 450, 290)
# 函数的返回值是更新的 mask, bgdModel, fgdModel
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:, :, np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()

exit(0)


# newmask is the mask image I manually labelled
newmask = cv2.imread('newmask.png',0)

# whereever it is marked white (sure foreground), change mask=1
# whereever it is marked black (sure background), change mask=0
mask[newmask==0] = 0
mask[newmask==255] = 1

mask, bgdModel, fgdModel = cv2.grabCut(img,mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

mask = np.where((mask==2)|(mask==0), 0,1).astype('uint8')
img = img*mask[:, :, np.newaxis]
plt.imshow(img)
plt.colorbar()
plt.show()