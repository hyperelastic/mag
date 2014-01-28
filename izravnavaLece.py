import cv2
import numpy as np

matr = np.loadtxt("paraParat/matrikaFotoaparata")
popac = np.loadtxt("paraParat/popacenjeFotoaparata")


slika = cv2.imread('slikeVzorec/000.jpg')
vis,  sir = slika.shape[:2]
novaMatr, regZanim = cv2.getOptimalNewCameraMatrix(matr, popac,(sir,vis),1,(sir,vis))


print(regZanim)

kartaX, kartaY = cv2.initUndistortRectifyMap(matr, popac, None,
                                                novaMatr, (sir,vis), 5)

slikaOdpac = cv2.remap(slika, kartaX, kartaY, cv2.INTER_LINEAR)
x,y,w,h = regZanim
slikaOdpac = slikaOdpac[y:y+h, x:x+w]

mala = cv2.resize(slikaOdpac, (0,0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
cv2.imshow('slikaOdpac', mala)
cv2.waitKey(4000)
cv2.destroyAllWindows()