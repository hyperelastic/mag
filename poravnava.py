########### PORAVNAVA STIRI###########
#   Zahteva paraParat/lokKodBodoce in paraParat/faktorPomanjsanjaSlike
#   Ta skripta najprej poravna prvo sliko glede na kode z znanimi fiz. polozaji,


import cv2
import numpy as np
from glob import glob
cv2.namedWindow('Prikaz', flags=cv2.WINDOW_KEEPRATIO)


def pokazi(slika, cas):
    cv2.imshow('Prikaz', slika)
    cv2.waitKey(cas)



#'''Tukaj vnesi ime mape, kamor spadajo izravnane slike'''
imeIzhodneMapeRel = './slikePoravnane/vzmet2'




lokKodBodoce = np.loadtxt("paraParat/lokacijeKod")
lokKodBodoce = np.array([lokKodBodoce[i] for i in [3, 4, 7, 8]])
merilo = np.loadtxt("paraParat/merilo")
fakPom = 0.2        #za grobo iskanje
vOko = 80           #velikost okolice kode za natancno iskanje

imenaSlikKod = sorted(glob('slikeKoda/[0-20]*.jpg'))
kode = []
kodeM = []
for i in [3, 4, 7, 8]:
    '''Shrani kode v imenika kode in kodeM'''
    slika = cv2.imread(imenaSlikKod[i], cv2.IMREAD_GRAYSCALE)
    slika = cv2.bilateralFilter(slika, 9, 75, 75)
    kode.append(slika)
    slika = cv2.resize(slika, (0, 0), fx = fakPom, fy=fakPom,
                        interpolation = cv2.INTER_AREA)
    kodeM.append(slika)

imenaSlik = sorted(glob('./slikeIzravnane/*.jpg')) 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
for j, iS in enumerate(imenaSlik[:]):
    sl = cv2.imread(iS); print(iS[-14:])
    slC = cv2.imread(iS, cv2.IMREAD_GRAYSCALE)
    slC = clahe.apply(slC)
    slC = cv2.bilateralFilter(slC, 9, 75, 75)
    #pokazi(slC, 500)
    slCM = cv2.resize(slC, (0,0), fx=fakPom, fy=fakPom,
                        interpolation=cv2.INTER_AREA)
    lokKodTrenutne = [] 
    for k, K in zip(kodeM, kode):
        slika = cv2.matchTemplate(slCM, k, cv2.TM_SQDIFF)
        void, void, lokPribl, void = cv2.minMaxLoc(slika)
        lokPribl = [(i+np.shape(k)[0]/2)/fakPom for i in lokPribl]
        slika = slC[lokPribl[1]-vOko:lokPribl[1]+vOko,
                        lokPribl[0]-vOko:lokPribl[0]+vOko]
        tmp = np.average(np.shape(slika))
        slika = cv2.matchTemplate(slika, K, cv2.TM_SQDIFF)
        void, void, lok, void = cv2.minMaxLoc(slika)
        lok = [ (lP+l+(np.average(np.shape(K))-tmp)/2)
                    for lP, l in zip(lokPribl, lok)]
        lokKodTrenutne.append(lok)
    lokKodBodoce = np.float32(lokKodBodoce).reshape(-1,1,2)
    lokKodTrenutne = np.float32(lokKodTrenutne).reshape(-1,1,2)
    M = cv2.getPerspectiveTransform(lokKodTrenutne, lokKodBodoce)

    #'''vztrajnost - vprasanje, koliko pomaga'''
    #if j!=0: 
    #    M = np.add(2.0*M, Mprej)/3.0
    #Mprej = M
        
    #'''prvo zvitje'''
    slC = cv2.warpPerspective(slC, M, 
                                (int(3698+120*merilo), int(2400+120*merilo)))

    #'''premik v okolici vpetja'''
    if j==0: cv2.imwrite("./slikePoravnane/osnovnaC.jpg", slC)
    slika = cv2.imread("./slikePoravnane/osnovnaC.jpg", cv2.IMREAD_GRAYSCALE)
    slika = slika[2420:-20,1200:1700]
    slika = cv2.matchTemplate(slC, slika, cv2.TM_SQDIFF)
    void, void, lokZac, void = cv2.minMaxLoc(slika)
    if abs(np.average([1200-lokZac[0], 2420-lokZac[1]]))>10.0:
        print("Nekaj ne stima... >(")
        cv2.imwrite(imeIzhodneMapeRel + '/' + iS[-14:-4] + 'NAP' + iS[-4:], sl)
    else:
        sl = cv2.warpPerspective(sl, M, 
                                    (int(3698+120*merilo), int(2400+120*merilo)))
        M1 = np.eye((3)); M1[0,2]=1200-lokZac[0]; M1[1,2] = 2420-lokZac[1]
        M1 = np.float32(M1[:2,:])
        sl = cv2.warpAffine(sl, M1,
                                (int(3698+120*merilo), int(2400+120*merilo)))
        cv2.imwrite(imeIzhodneMapeRel + '/' + iS[-14:], sl)
    

cv2.destroyAllWindows()







