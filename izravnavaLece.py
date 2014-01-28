import cv2
import numpy as np
from glob import glob

matr = np.loadtxt("paraParat/matrikaFotoaparata")
popac = np.loadtxt("paraParat/popacenjeFotoaparata")


slika = cv2.imread('slikeVzorec/000.jpg')
vis,  sir = slika.shape[:2]
novaMatr, regZanim = cv2.getOptimalNewCameraMatrix(matr, popac,(sir,vis),1,(sir,vis))


kartaX, kartaY = cv2.initUndistortRectifyMap(matr, popac, None,
                                                novaMatr, (sir,vis), 5)




slike = glob('slikeVzorec/*.jpg')
x,y,w,h = regZanim
for sl in slike:
    print("Izravnavam sliko " + sl[-7:])
    slikaOdpac = cv2.remap(cv2.imread(sl), kartaX, kartaY, cv2.INTER_LINEAR)
    slikaOdpac = slikaOdpac[y:y+h, x:x+w]
    cv2.imwrite("slikeIzravnane/" + sl[-7:], slikaOdpac)
#     mala = cv2.resize(slikaOdpac, (0,0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
#     cv2.imshow('slikaOdpac', mala)
#     cv2.waitKey(400)
#     cv2.destroyAllWindows()