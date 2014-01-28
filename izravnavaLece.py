import cv2
import numpy as np
from glob import glob
from PIL import Image
from PIL.ExifTags import TAGS


def casZajema(imeDat):
    ''' Vrne string s casom zajema hhmmss'''
    slikaPil = Image.open(imeDat)
    exif = {TAGS[k]: v for k, v in slikaPil._getexif().items() if k in TAGS}
    return(exif['DateTimeOriginal'][-8:].translate(None, ':'))


matr = np.loadtxt("paraParat/matrikaFotoaparata")
popac = np.loadtxt("paraParat/popacenjeFotoaparata")

slika = cv2.imread('slikeVzorec/000.jpg')
vis,  sir = slika.shape[:2]
novaMatr, regZanim = cv2.getOptimalNewCameraMatrix(matr, popac,(sir,vis),1,(sir,vis))

kartaX, kartaY = cv2.initUndistortRectifyMap(matr, popac, None,
                                                novaMatr, (sir,vis), 5)

slike = sorted(glob('slikeVzorec/*.jpg'))
x,y,w,h = regZanim
for sl in slike:
    imeIzravnane = sl[-7:-4] + "-" + casZajema(sl) + ".jpg"
    print("Izravnavam sliko " + imeIzravnane)
    slikaOdpac = cv2.remap(cv2.imread(sl), kartaX, kartaY, cv2.INTER_LINEAR)
    slikaOdpac = slikaOdpac[y:y+h, x:x+w]
    cv2.imwrite("slikeIzravnane/" + imeIzravnane, slikaOdpac)
#     mala = cv2.resize(slikaOdpac, (0,0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
#     cv2.imshow('slikaOdpac', mala)
#     cv2.waitKey(400)
# #     cv2.destroyAllWindows()