import cv2
import numpy as np


slC = cv2.imread("./slikePoravnane/nelin/465-150144.jpg", cv2.IMREAD_GRAYSCALE)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe.apply(slC)
#slC = cv2.bilateralFilter(slC, 5, 175, 175)
slC = cv2.medianBlur(slC, 9)
#cv2.namedWindow("Prikaz", cv2.WINDOW_NORMAL)
cv2.imshow("Prikaz", slC[1500:,1100:])
#slika = slC[2300:-50, 1650:1660]
#cv2.imshow("Prikaz", slika)
cv2.waitKey(10000)
cv2.destroyAllWindows()




