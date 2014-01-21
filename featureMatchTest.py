# Tukaj testiram test značilk za eno kodo.

import cv2
import numpy as np


slika = cv2.imread('slikeKoda/niPoVrsti04.jpg', cv2.IMREAD_GRAYSCALE)
# slika = cv2.GaussianBlur( slika, (5, 5), 0 );
vogali = cv2.goodFeaturesToTrack(slika, 10, 0.01, 10)
vogali = np.int0(vogali)

for v in vogali:
    x,y = v.ravel()
    cv2.circle(slika,(x,y),3,255,-1)

#imshow()
cv2.namedWindow("Slikca1", cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()
cv2.imshow("Slikca1", slika)
cv2.waitKey(3000)
cv2.destroyWindow("Slikca1")