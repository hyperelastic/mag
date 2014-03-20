import cv2
import numpy as np
from operator import add


sl = cv2.imread("./slikePoravnane/nelin/464-150143.jpg")
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
void, void, lMaxKanal1, void = cv2.minMaxLoc(pas)
lMaxKanal1 = [lMaxKanal1[0]+120, lMaxKanal1[1]+1550]


cv2.namedWindow("Prikaz", cv2.WINDOW_NORMAL)
for i in range(9):
    print(lMaxKanal1)
    radij = 15
    okence = kanal1[lMaxKanal1[1]-2*radij:lMaxKanal1[1]+2*radij+1,
                        lMaxKanal1[0]-2*radij:lMaxKanal1[0]+2*radij+1]
    okence = np.fliplr(okence)
    okence = cv2.linearPolar(okence, (2*radij, 2*radij), int(1.1*radij),
                                cv2.INTER_LINEAR)
    pas = cv2.GaussianBlur(okence[radij:-radij,-.2*radij-1:], (5,15), 0)
    cv2.imshow("Prikaz", pas)
    cv2.waitKey(2000)
    void, void, lMaxOkence, void = cv2.minMaxLoc(pas)
    okence = cv2.circle(okence, 
                        tuple([4*radij, lMaxOkence[1]+radij])
                        , 0, 255, thickness=2)
    print(lMaxOkence)
    kot = 0.5*np.pi*(lMaxOkence[1]-radij)/(radij)
    print(kot)
    lMaxKanal1 = [lMaxKanal1[0] + radij*np.cos(kot),
                    lMaxKanal1[1] - radij*np.sin(kot)]


    kanal1 = cv2.circle(kanal1, tuple([int(l) for l in lMaxKanal1]),
                            0, 255, thickness=2)

    cv2.imshow("Prikaz", okence)
    cv2.waitKey(10000)
#kanal1 = cv2.circle(kanal1, tuple(lMaxKanal1), 0, 255, thickness=2)
cv2.imshow("Prikaz", kanal1[-500:,:])
#cv2.imshow("Prikaz", sl)
cv2.waitKey(20000)


cv2.destroyAllWindows()




#void, slC = cv2.threshold(slC, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
