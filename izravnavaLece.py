import cv2
import numpy as np
from glob import glob
from PIL import Image
from PIL.ExifTags import TAGS
from re import sub

def casZajema(imeDat):
    ''' Vrne string s casom zajema hhmmss'''
    slikaPil = Image.open(imeDat)
    exif = {TAGS[k]: v for k, v in slikaPil._getexif().items() if k in TAGS}
    print(exif['DateTimeOriginal'][-8:])
    return(sub(':', '', exif['DateTimeOriginal'][-8:]))




#'''Tukaj vnesi ime mape, od kjer naj izravnam slike'''
imeMapeAbs = '/home/klmn/SlikeMag/vzmet2'
koncnicaDatotek = '.JPG'




matr = np.loadtxt("paraParat/matrikaFotoaparata")
popac = np.loadtxt("paraParat/popacenjeFotoaparata")

slika = cv2.imread('slikeVzorec/000.jpg')
vis,  sir = slika.shape[:2]
novaMatr, regZanim = cv2.getOptimalNewCameraMatrix(matr, popac,(sir,vis),1,(sir,vis))

kartaX, kartaY = cv2.initUndistortRectifyMap(matr, popac, None,
                                                novaMatr, (sir,vis), 5)

slike = sorted(glob(imeMapeAbs + '/*' + koncnicaDatotek))
x,y,w,h = regZanim
for j, sl in enumerate(slike):
    imeIzravnane = ("%03d" % j) + "-" + casZajema(sl) + ".jpg"
    print("Izravnavam sliko " + imeIzravnane)
    slikaOdpac = cv2.remap(cv2.imread(sl), kartaX, kartaY, cv2.INTER_LINEAR)
    slikaOdpac = slikaOdpac[y:y+h, x:x+w]
    cv2.imwrite("slikeIzravnane/" + imeIzravnane, slikaOdpac)
