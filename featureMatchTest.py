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



ujemi = sorted(ujemi, key = lambda x:x.distance)
dobri = ujemi[:15]

notXY = np.float32([ t1[d.queryIdx].pt for d in dobri ]).reshape(-1,1,2)
venXY = np.float32([ t2[d.trainIdx].pt for d in dobri ]).reshape(-1,1,2)

M, mask = cv2.findHomography(notXY, venXY, cv2.LMEDS, 5.0)

print(notXY - venXY)

print(M)

h,w = sl1.shape
notSR = np.float32([[h/2+1, w/2+2]]).reshape(-1,1,2)
venSR = cv2.perspectiveTransform(notSR, M)
print(notSR)
print(venSR[0][0])

cv2.circle(sl1, (int(notSR[0][0][0]), int(notSR[0][0][1])), 10, (200, 200, 200), 2)
cv2.circle(sl2, (int(venSR[0][0][0]), int(venSR[0][0][1])), 10, (200, 200, 200), 2)

sl3 = cv2.drawMatches(sl1,t1,sl2,t2,dobri, flags=2)

#imshow()
cv2.namedWindow("Slikca1", cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()
cv2.imshow("Slikca1", sl3)
cv2.waitKey(1000)
cv2.destroyWindow("Slikca1")