#Tukaj testiram, ali bi se dalo izboljsat poravnanje s pomocjo iskanja krogov v kodah
# Tukaj testiram test značilk za eno kodo.

import cv
import cv2
import numpy as np


#iskanje krogov
im = cv2.imread('slikeKoda/niPoVrsti01.jpg', cv2.IMREAD_GRAYSCALE)
im = cv2.GaussianBlur( im, (5, 5), 0 );
krogi = cv2.HoughCircles( im, cv.CV_HOUGH_GRADIENT, 5, 5 )
if krogi != None:
    for k in krogi[0]:
        cv2.circle(im, (k[0], k[1]), k[2], 255)
        print k
else: print("Nisem nasel krogov!")



#imshow()
cv2.namedWindow("Slikca1", cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()
cv2.imshow("Slikca1", im)
cv2.waitKey(3000)
cv2.destroyWindow("Slikca1")