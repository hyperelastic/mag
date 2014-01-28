# Tukaj testiram test značilk za eno kodo.

import cv2
import numpy as np


sl1 = cv2.imread('slikeKoda/niPoVrsti02.jpg', cv2.IMREAD_GRAYSCALE)
sl2 = cv2.imread('slikeKoda/testnaZaFeatureMatch.jpg', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB()
t1, op1 = orb.detectAndCompute(sl1,None)
t2, op2 = orb.detectAndCompute(sl2,None)

naj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
ujemi = naj.match(op1,op2)
matches = sorted(ujemi, key = lambda x:x.distance)

for m in matches[:15]:
    print(m)

sl3 = cv2.drawMatches(sl1,t1,sl2,t2,matches[:15], flags=2)


#imshow()
cv2.namedWindow("Slikca1", cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()
cv2.imshow("Slikca1", sl3)
cv2.waitKey(5000)
cv2.destroyWindow("Slikca1")