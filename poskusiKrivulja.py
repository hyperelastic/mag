import cv2
import numpy as np
from operator import add


sl = cv2.imread("./slikePoravnane/nelin/016-145511.jpg")
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6,6))

sl = cv2.cvtColor(sl, cv2.COLOR_BGR2HSV)
sl = cv2.insertChannel(clahe.apply(cv2.extractChannel(sl, 2)), sl, 2)
sl = cv2.cvtColor(sl, cv2.COLOR_HSV2BGR)
sl = cv2.medianBlur(sl, 7)


kanal1 = cv2.extractChannel(sl, 1)
visIzreza = 800; sirIzreza = 1500
kanal1 = kanal1[visIzreza:-50, sirIzreza:-160]
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(102,102))
kanal1 = clahe.apply(kanal1)

pas = kanal1[1550:-70, 120:130]
pas = cv2.GaussianBlur(pas, (51,21), 0)
void, void, lMax, void = cv2.minMaxLoc(pas)
lMax = [lMax[0]+120, lMax[1]+1550]
print(lMax[1])

zihSpredaj = zihZadaj = kanal1*0
zihSpredaj = cv2.circle(zihSpredaj, tuple(lMax), 0, 255, thickness=2)

void, zihZadaj = cv2.threshold(kanal1, 1, 255,
                                cv2.THRESH_BINARY+cv2.THRESH_OTSU)
zihZadaj = cv2.circle(zihZadaj, tuple(map(add, lMax, [-60, -4])),
                        2, 255, thickness=5)
zihZadaj = cv2.circle(zihZadaj, tuple(map(add, lMax, [-60, 15])),
                        2, 255, thickness=5)

zihNevem = cv2.subtract(zihZadaj, zihSpredaj)

void, markerji = cv2.connectedComponents(zihSpredaj)
markerji = markerji+1
markerji[zihNevem==255] = 0
markerji = cv2.watershed(kanal1, markerji)

kanal1[markerji==-1] = 255



#kanal1 = cv2.circle(kanal1, tuple(lMax), 0, 255, thickness=2)
#kanal1 = cv2.circle(kanal1, tuple(map(add, lMax, [-60, -4])),
#                        2, 255, thickness=5)
#kanal1 = cv2.circle(kanal1, tuple(map(add, lMax, [-60, 15])),
#                        2, 255, thickness=5)
#cv2.imshow("Prikaz", kanal1[-500:,:])

#cv2.namedWindow("Prikaz", cv2.WINDOW_NORMAL)
#cv2.imshow("Prikaz", pas)
cv2.imshow("Prikaz", zihNevem[-500:,:])


#cv2.imshow("Prikaz", sl)
cv2.waitKey(10000)

cv2.destroyAllWindows()




#void, slC = cv2.threshold(slC, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
