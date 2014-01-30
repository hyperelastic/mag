########### PORAVNAVA ###########
#   Zahteva paraParat/lokacijeKod in paraParat/faktorPomanjsanjaSlike
#   Ta skripta najprej poravna prvo sliko glede na kode z znanimi fiz. polozaji,

import cv2
import numpy as np
from glob import glob

lokacijeKod = np.loadtxt("paraParat/lokacijeKod")
fakPom = np.loadtxt("paraParat/faktorPomanjsanjaSlike")

imenaSlik = sorted(glob('slikeVzorec/*.jpg'))

for iS in imenaSlik[:2]:
    print(iS)
    sl = cv2.imread(iS)
    



# 
# 
# 
# imePrve = sorted(glob('slikeVzorec/*.jpg'))[7]
# fakPomanj = 0.4     #faktor pomanjsanja slike za grobo iskanje
# merilo = 1.5        #merilo slikovneDimenzije/fizikalneDimenzije
# vOko = 70           #velikost okolice kode za natancno iskanje
# 
# 
# 
# def izboljsajPolozajKode(koda, okolica):
#     ''' Vrne kodo z izboljsanim polozajem '''
#     
#     orb = cv2.ORB(50, 2, 5)
#     t1, op1 = orb.detectAndCompute(okolica[0],None)
#     t2, op2 = orb.detectAndCompute(koda[0],None)
#     naj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     ujemi = naj.match(op1,op2)
#     ujemi = sorted(ujemi, key = lambda x:x.distance)
# 
#     notXY = np.array([ t1[d.queryIdx].pt for d in ujemi[:3] ])
#     venXY = np.array([ t2[d.trainIdx].pt for d in ujemi[:3] ])
#     premik = np.average(notXY-venXY, axis=0)
#     stDevPremik = np.std(notXY-venXY, axis=0)
#     
#     delta = premik - (np.array(okolica[0].shape) - np.array(koda[0].shape))/2
#     koda[2] = [x+d for x, d in zip(koda[2], delta)]
#     
#     print("premik"); print(premik)
#     print("stDevPremik"); print(stDevPremik)
#     print("delta"); print(delta)
#     print("\n")
#     srKodeNaKodi = [x/2 for x in koda[0].shape]
#     srKodeNaOkolici = tuple([sr+d for sr, d in zip(srKodeNaKodi, premik)])
#     cv2.circle(okolica[0], (int(srKodeNaOkolici[0]),
#                                 int(srKodeNaOkolici[1])),
#                                     8, 255, 1, cv2.LINE_AA)
# 
#     sl3 = cv2.drawMatches(okolica[0], t1, koda[0], t2, ujemi[:3], 200, flags=2)
#     cv2.imshow("Izboljsava", sl3)
#     cv2.waitKey(500)
#     
#     return(koda)
# 
# 
# 
# def fizKoordinateVogalov(merilo, zamikX, zamikY):
#     Tocke = np.array([  [0.0, 1500.0],
#                         [2361.0, 1500.0],
#                         [2365.0, 0.0],
#                         [0.0, 0.0]   ])
# 
#     Tocke[:,1] = Tocke[:,1]+zamikX
#     Tocke[:,0] = Tocke[:,0]+zamikY
#     return(np.float32(Tocke*merilo).reshape(-1,1,2))
# 
# 
# 
# prva = cv2.imread(imePrve, cv2.IMREAD_GRAYSCALE)
# prvaMala = cv2.resize(prva, (0,0), fx=fakPomanj, fy=fakPomanj, interpolation=cv2.INTER_AREA)
# 
# 
# 
# kode = []
# kodeMale = []
# imenaSlikKod = sorted(glob('slikeKoda/[0-20]*.jpg'))
# for iSK in imenaSlikKod:
#     koda = cv2.imread(iSK, cv2.IMREAD_GRAYSCALE)
#     kode.append([koda, int(iSK[-6:-4]), [0, 0]])
#     koda = cv2.resize(koda, (0,0), fx=fakPomanj, fy=fakPomanj, interpolation=cv2.INTER_AREA)
#     kodeMale.append([koda, int(iSK[-6:-4]), [0, 0]])
# 
# 
# 
# for koda, kodaMala in zip(kode, kodeMale):
#     res = cv2.matchTemplate(prvaMala, kodaMala[0], cv2.TM_SQDIFF)
#     void, void, zgorajLevo, void = cv2.minMaxLoc(res)
#     sredinaMala = [int(x/2+y) for (x, y) in zip(kodaMala[0].shape, zgorajLevo)]
# #     sredinaMala = [x+y for x, y in zip(sredinaMala, [5, 10])]     # ZA TEST DELOVANJA IZBOLJSAVEPOLOZAJA
#     kodaMala[2] = [int(s) for s in sredinaMala]
#     koda[2] = [int(s/fakPomanj) for s in sredinaMala]
#     cv2.putText(prvaMala, str(kodaMala[1]), tuple(kodaMala[2]),
#                                 cv2.FONT_HERSHEY_SIMPLEX, fakPomanj, 255)
# 
# 
# 
# kodeRobovi = [kode[i] for i in [3,4,7,8]]
# kodeRoboviMale = [kodeMale[i] for i in [3,4,7,8]]
# 
# 
# 
# okolice = []
# for koda in kodeRobovi:
#     xSr, ySr = koda[2][0], koda[2][1]
#     okolice.append([prva[ySr-vOko:ySr+vOko, xSr-vOko:xSr+vOko], [xSr, ySr]])
# 
# 
# 
# vhodniVogali = []
# for okolica, koda, kodaMala in zip(okolice, kodeRobovi, kodeRoboviMale):
#     koda = izboljsajPolozajKode(koda, okolica)
#     cv2.destroyAllWindows()
#     vhodniVogali.append(koda[2])
#     kodaMala[2] = [int(k*fakPomanj) for k in koda[2]]
# vhodniVogali = np.float32(vhodniVogali).reshape(-1,1,2)
# izhodniVogali = fizKoordinateVogalov(merilo, 120.0, 120.0)
# 
# 
# 
# M = cv2.getPerspectiveTransform(vhodniVogali, izhodniVogali)
# 
# 
# 
# vhodneKoordinateVseh = []
# for koda in kode:
#     vhodneKoordinateVseh.append(koda[2])
# vhodneKoordinateVseh = np.float32(vhodneKoordinateVseh).reshape(-1,1,2)
# fizKoordinateVseh = cv2.perspectiveTransform(vhodneKoordinateVseh, M)
# print fizKoordinateVseh[:,0,:]
# 
# 
# 
# prvaNova = cv2.warpPerspective(prva, M, (int(3698+120*merilo), int(2400+120*merilo)))
# for fk in fizKoordinateVseh[:, 0, :]:
#     cv2.circle(prvaNova, tuple(fk), 8, 255, 1, cv2.LINE_AA)
# 
# 
# 
# cv2.imwrite("prvaNova.jpg", prvaNova)
# 
# 
# np.savetxt("paraParat/lokacijeKod", fizKoordinateVseh[:,0,:])
# np.savetxt("paraParat/faktorPomanjsanjaSlike", [fakPomanj])
# 
# 
# 












