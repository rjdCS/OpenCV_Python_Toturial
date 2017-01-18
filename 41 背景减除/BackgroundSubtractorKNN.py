"""
"""

import numpy as np
import cv2

PATH = '../images/'

cap = cv2.VideoCapture(PATH+'vtest.avi')

fgbg = cv2.createBackgroundSubtractorKNN()

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