

import numpy as np
import cv2

PATH = '../images/'

cap = cv2.VideoCapture(PATH+'vtest.avi')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

fgbg = cv2.createBackgroundSubtractorGMG()

while(1):
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()