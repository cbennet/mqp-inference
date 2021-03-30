import cv2
import numpy as np
import math
import time
distInch = 48/1133

cv2.namedWindow("preview")
cap = cv2.VideoCapture(2)
idStore = {}
time.sleep(1)
while(1):
    _, image = cap.read()
    cv2.imshow('preview', image)
    cv2.waitKey(20)