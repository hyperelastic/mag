########### PORAVNAVA PRVE SLIKE ###########
#Kode morajo bit lepo centrirane.
#Prva slika naj bo taka, da so lepo vidne vse kode.
#Ta skripta najprej poravna prvo sliko glede na kode z znanimi polozaji,
#nato pa poisce se polozaje drugih kod in vse skupaj shrani v tekst.

import cv2
import numpy as np
from glob import glob



imePrve = sorted(glob('slikeVzorec/*.jpg'))[7]
fakPomanj = 0.4
vOko = 70 #velikost okolice



def izboljsajPolozajKode(koda, okolica):
    ''' Vrne kodo z izboljsanim polozajem '''
    
    orb = cv2.ORB()
    t1, op1 = orb.detectAndCompute(okolica[0],None)
    t2, op2 = orb.detectAndCompute(koda[0],None)
    naj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ujemi = naj.match(op1,op2)
    ujemi = sorted(ujemi, key = lambda x:x.distance)

    notXY = np.float32([ t1[d.queryIdx].pt for d in ujemi[:30] ]).reshape(-1,1,2)
    venXY = np.float32([ t2[d.trainIdx].pt for d in ujemi[:30] ]).reshape(-1,1,2)
    M, void = cv2.findHomography(notXY, venXY, cv2.RANSAC, 1.0)
    Minv = np.linalg.inv(M)
    
    srKodeNaKodi = np.float32([x/2 for x in koda[0].shape]).reshape(-1,1,2)
    srKodeNaOkolici = cv2.perspectiveTransform(srKodeNaKodi, Minv)
    
    delta = [po - pred/2 for (pred, po) in zip(okolica[0].shape, srKodeNaOkolici[0][0])]
    print(delta)
    
    cv2.circle(okolica[0], (int(srKodeNaOkolici[0][0][0]), int(srKodeNaOkolici[0][0][1])), 3, (250, 250, 100), 2)

    sl3 = cv2.drawMatches(okolica[0], t1, koda[0], t2, ujemi[:3], 200, flags=2)
    cv2.imshow("Izboljsava", sl3)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    return(koda)




prva = cv2.imread(imePrve, cv2.IMREAD_GRAYSCALE)
prvaMala = cv2.resize(prva, (0,0), fx=fakPomanj, fy=fakPomanj, interpolation=cv2.INTER_AREA)

kode = []
kodeMale = []
imenaSlikKod = sorted(glob('slikeKoda/[0-20]*.jpg'))
for iSK in imenaSlikKod: #kode[i] = [koda, stKode, [polozX, polozY]]
    koda = cv2.imread(iSK, cv2.IMREAD_GRAYSCALE)
    kode.append([koda, int(iSK[-6:-4]), [0, 0]])
    koda = cv2.resize(koda, (0,0), fx=fakPomanj, fy=fakPomanj, interpolation=cv2.INTER_AREA)
    kodeMale.append([koda, int(iSK[-6:-4]), [0, 0]])

for koda, kodaMala in zip(kode, kodeMale):
    res = cv2.matchTemplate(prvaMala, kodaMala[0], cv2.TM_SQDIFF)
    void, void, zgorajLevo, void = cv2.minMaxLoc(res)
    sredinaMala = [int(x/2+y+1) for (x, y) in zip(kodaMala[0].shape, zgorajLevo)] ##!!!!!! +3 !!!!!! test
    kodaMala[2] = [int(s) for s in sredinaMala]
    koda[2] = [int(s/fakPomanj) for s in sredinaMala]
#     cv2.circle(prvaMala, tuple(kodaMala[2]), 12, 255, 1, cv2.LINE_AA)
    cv2.putText(prvaMala, str(kodaMala[1]), tuple(kodaMala[2]),
                                cv2.FONT_HERSHEY_SIMPLEX, fakPomanj, 255)


kode = kode[3:11]
kodeMale = kodeMale[3:11]
okolice = []
for koda in kode:
    xSr, ySr = koda[2][0], koda[2][1]
    okolice.append([prva[ySr-vOko:ySr+vOko, xSr-vOko:xSr+vOko], [xSr, ySr]])

for okolica, koda, kodaMala in zip(okolice, kode, kodeMale):
    print(koda[1])
    koda = izboljsajPolozajKode(koda, okolica)
    kodaMala[2] = [int(k*fakPomanj) for k in koda[2]]

for kodaMala in kodeMale:
    cv2.circle(prvaMala, tuple(kodaMala[2]), 3, 255, 1, cv2.LINE_AA)
    
# 
# cv2.imshow('slika', prvaMala)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()




# class Slika:
#     ''' Razred s crnobelo in barvno razlicico slike '''
#     zeVnesena = False
#     def __init__(self, imeDatoteke):
#         if Slika.zeVnesena == False:
#             Slika.original = cv2.imread(imeDatoteke)
#             Slika.siva = cv2.imread(imeDatoteke, cv2.IMREAD_GRAYSCALE)
#             Slika.dimenzije = np.shape(Slika.siva)
#             Slika.zeVnesena = True
#         else:
#             pass
#     
#     def prepisiSliko(self, imeDatoteke):
#         print('Prepisujem sliko z datoteko ' + imeDatoteke)
#         Slika.original = cv2.imread(imeDatoteke)
#         Slika.siva = cv2.imread(imeDatoteke, cv2.IMREAD_GRAYSCALE)
# 
# 
# class Izrez(Slika):
#     ''' Razred z metodami za izrez slike.'''
#     def __init__(self, imeSlike):
#         Slika.__init__(self, imeSlike)
#     
#     def izrezi(self, sredisceX, sredisceY, dim):
#         self.mejeY = [int(sredisceY)-dim, int(sredisceY)+dim]
#         self.mejeX = [int(sredisceX)-dim, int(sredisceX)+dim]
#         if self.mejeY[0] < 0: self.mejeY[0]=0
#         if self.mejeY[1] > Slika.dimenzije[0]-1: self.mejeY[1]=Slika.dimenzije[0]-1
#         if self.mejeX[0] < 0: self.mejeX[0]=0
#         if self.mejeX[1] > Slika.dimenzije[1]-1: self.mejeX[1]=Slika.dimenzije[1]-1
#         self.siviIzrez = Slika.siva[self.mejeY[0]:self.mejeY[1],
#                                 self.mejeX[0]:self.mejeX[1]]
#     
#     def prepisiIzrez(self, imeDatoteke):
#         Slika.prepisiSliko(imeDatoteke)
#         self.siviIzrez = Slika.siva[self.mejeY[0]:self.mejeY[1],
#                                     self.mejeX[0]:self.mejeX[1]]
#     
#     def prikaziIzrez(self):
#         tmp = Slika.siva
#         cv2.rectangle(tmp, (self.mejeX[0], self.mejeY[0]),
#                                 (self.mejeX[1], self.mejeY[1]), 0, 10)
#         tmp = cv2.resize(tmp, tuple(int(x/3) for x in np.shape(tmp)[::-1]))
#         cv2.imshow('izrez', tmp)
#         cv2.waitKey(0)
# 
# 
# class KodaVsliki(Koda, Izrez):
#     ''' Razred s slikovno kodo in metodami za njeno lociranje v sliki. '''
# 
#     def __init__(self, imeDatKode, dolzinaStraniceKode, imeDatSlike):
#         Koda.__init__(self, imeDatKode, dolzinaStraniceKode)
#         Izrez.__init__(self, imeDatSlike)
#         
#     def lociranjePovsod(self, oknoPodPix):
#         ujem = cv2.chTemplate(Slika.siva, self.skrcenaKoda, cv.CV_TM_SQDIFF_NORMED)
#         self.ocena, _,lokOkna ,_ = cv2.minMaxLoc(ujem)
#         vsotaFaktorjev = 0.
#         self.sredisce = np.array([0., 0.])
#         for i in range(lokOkna[1]-oknoPodPix, lokOkna[1]+oknoPodPix+1):
#             for j in range(lokOkna[0]-oknoPodPix, lokOkna[0]+oknoPodPix+1):
#                 vsotaFaktorjev += (1-ujem[i, j])**2
#                 self.sredisce += np.array([i, j])*(1-ujem[i, j])**2
#         self.sredisce = (self.sredisce/vsotaFaktorjev)[::-1]
#         self.sredisce += self.sredisceKode
# 
#     def lociranjeIzrez(self, oknoPodPix, sredisceY, sredisceX, dim, originalnaOcena):
#         
#         Izrez.izrezi(self, sredisceY, sredisceX, dim)
#         ujem = cv2.matchTemplate(self.siviIzrez, self.skrcenaKoda, cv.CV_TM_SQDIFF_NORMED)
#         ocena, _,lokOkna ,_ = cv2.minMaxLoc(ujem)
#         print(originalnaOcena, ocena)
#         vsotaFaktorjev = 0.
#         self.sredisce = np.array([0., 0.])
#         if ocena < 10.5*originalnaOcena:
#             for i in range(lokOkna[1]-oknoPodPix, lokOkna[1]+oknoPodPix+1):
#                 for j in range(lokOkna[0]-oknoPodPix, lokOkna[0]+oknoPodPix+1):
#                     vsotaFaktorjev += (1-ujem[i, j])**2
#                     self.sredisce += np.array([i, j])*(1-ujem[i, j])**2
#             self.sredisce = (self.sredisce/vsotaFaktorjev)[::-1]
#             self.sredisce = list(x+y+z-dim for x, y, z in
#                                         zip(self.sredisce, self.sredisceKode,
#                                                 [sredisceY, sredisceX]))
#             self.sredisce = np.array(self.sredisce)
#     
#     def natisniSredisce(self):
#         print(self.sredisce)
# 
# 
# def zapisiFizKoordinateKod(koef, razdX, razdY, zamikX, zamikY):
#     Tocke = []
#     Tocke.append([0.0, 3*razdY])
#     for j in range(3, -1, -1):
#         Tocke.append([razdX, razdY*j])
#     for j in range(0, 3):
#         Tocke.append([0.0, razdY*j])
#     Tocke = np.array(Tocke)*koef
#     Tocke[:,1] = Tocke[:,1]+zamikX
#     Tocke[:,0] = Tocke[:,0]+zamikY
#     return(np.array(Tocke)*koef)
# 
# 
# kode1 = []
# for i in range(3,11):
#     kode1.append(KodaVsliki("kodaPNG/" + str(i) + ".png", 63, 'test1.jpg'))
# for k in kode1:
#     k.lociranjePovsod(3)
# kode = kode1[:]
# 
# potVhod = "slike/neobdelane/linearniPoskus/odpacene/"
# potIzhod = "slike/poravnane/linearniPoskus/"
# 
# 
# 
# for i in range(0, 40):
#     print(str(i))
#     
#     kode[0].prepisiSliko(potVhod + str(i) + ".jpg")
#     for k1, k in zip(kode1, kode):
#         k.lociranjeIzrez(3, k1.sredisce[0], k1.sredisce[1], 120, k1.ocena)
#         
#     
#     fiz = zapisiFizKoordinateKod(1.0, 2363.0, 500.0, 100.0, 100.0)
#     izvor = []
#     for k in kode:
#         izvor.append(k.sredisce)
#     izvor = np.array(izvor)
# 
#     print(izvor)
#     print("\n")
#     
#     M, _ = cv2.findHomography(izvor, fiz, cv.CV_LMEDS)
#     tmp = cv2.warpPerspective(Slika.original, M, 
#                             (int(np.max(fiz[:,0]))+100, int(np.max(fiz[:,1]))+100))
#     cv2.imwrite(potIzhod + str(i) + ".jpg", tmp)

