# Tukaj testiram test značilk za eno kodo.

import cv2
import numpy as np



im = cv2.imread('slikeKoda/niPoVrsti04.jpg', cv2.IMREAD_GRAYSCALE)
imNP = np.float32(im)
dst = cv2.cornerHarris(imNP, 3, 3, 0.04)
dst = cv2.dilate(dst, None)
im[dst>0.6*dst.max()] = 255


#imshow()
cv2.namedWindow("Slikca1", cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()
cv2.imshow("Slikca1", im)
cv2.waitKey(3000)
cv2.destroyWindow("Slikca1")