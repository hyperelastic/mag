########### PORAVNAVA PRVE SLIKE ###########
#   Kode morajo bit lepo centrirane.
#   Prva slika naj bo taka, da so lepo vidne vse kode.
#   Ta skripta najprej poravna prvo sliko glede na kode z znanimi fiz. polozaji,
#nato s homografijo oceni se fiz. polozaje drugih kod in vse skupaj shrani v
#besedilno datoteko.

import cv2
import numpy as np
from glob import glob



imePrve = sorted(glob('slikeVzorec/*.jpg'))[7]
fakPomanj = 0.4     #faktor pomanjsanja slike za grobo iskanje
merilo = 1.5        #merilo slikovneDimenzije/fizikalneDimenzije
vOko = 70           #velikost okolice kode za natancno iskanje



def izboljsajPolozajKode(koda, okolica):
    ''' Vrne kodo z izboljsanim polozajem '''
    
    orb = cv2.ORB()
    t1, op1 = orb.detectAndCompute(okolica[0],None)
    t2, op2 = orb.detectAndCompute(koda[0],None)
    naj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ujemi = naj.match(op1,op2)
    ujemi = sorted(ujemi, key = lambda x:x.distance)

    notXY = np.array([ t1[d.queryIdx].pt for d in ujemi[:5] ])
    venXY = np.array([ t2[d.trainIdx].pt for d in ujemi[:5] ])
    premik = np.average(notXY-venXY, axis=0)
    stDevPremik = np.std(notXY-venXY, axis=0)
    
    delta = premik - (np.array(okolica[0].shape) - np.array(koda[0].shape))/2
    koda[2] = [x+d for x, d in zip(koda[2], delta)]
    
#     print(premik)
#     print(stDevPremik)
#     print(delta)
#     print("\n")
#     srKodeNaKodi = [x/2 for x in koda[0].shape]
#     srKodeNaOkolici = tuple([sr+d for sr, d in zip(srKodeNaKodi, delta)])
#     cv2.circle(okolica[0], (int(srKodeNaOkolici[0]),
#                                 int(srKodeNaOkolici[1])),
#                                     3, 255, 2)
# 
#     sl3 = cv2.drawMatches(okolica[0], t1, koda[0], t2, ujemi[:5], 200, flags=2)
#     cv2.imshow("Izboljsava", sl3)
#     cv2.waitKey(500)
    
    return(koda)


def zapisiFizKoordinateKod(merilo, razdX, razdY, zamikX, zamikY):
    Tocke = []
    Tocke.append([0.0, 3*razdY])
    for j in range(3, -1, -1):
        Tocke.append([razdX, razdY*j])
    for j in range(0, 3):
        Tocke.append([0.0, razdY*j])
    Tocke = np.array(Tocke)
    Tocke[:,1] = Tocke[:,1]+zamikX
    Tocke[:,0] = Tocke[:,0]+zamikY
    return(np.float32(Tocke*merilo).reshape(-1,1,2))



prva = cv2.imread(imePrve, cv2.IMREAD_GRAYSCALE)
prvaMala = cv2.resize(prva, (0,0), fx=fakPomanj, fy=fakPomanj, interpolation=cv2.INTER_AREA)

kode = []
kodeMale = []
imenaSlikKod = sorted(glob('slikeKoda/[0-20]*.jpg'))
for iSK in imenaSlikKod:    #kode[i] = [koda, stKode, [polozX, polozY]]
    koda = cv2.imread(iSK, cv2.IMREAD_GRAYSCALE)
    kode.append([koda, int(iSK[-6:-4]), [0, 0]])
    koda = cv2.resize(koda, (0,0), fx=fakPomanj, fy=fakPomanj, interpolation=cv2.INTER_AREA)
    kodeMale.append([koda, int(iSK[-6:-4]), [0, 0]])

for koda, kodaMala in zip(kode, kodeMale):
    res = cv2.matchTemplate(prvaMala, kodaMala[0], cv2.TM_SQDIFF)
    void, void, zgorajLevo, void = cv2.minMaxLoc(res)
    sredinaMala = [int(x/2+y) for (x, y) in zip(kodaMala[0].shape, zgorajLevo)]
#     sredinaMala = [x+y for x, y in zip(sredinaMala, [5, 10])]     # ZA TEST DELOVANJA IZBOLJSAVEPOLOZAJA
    kodaMala[2] = [int(s) for s in sredinaMala]
    koda[2] = [int(s/fakPomanj) for s in sredinaMala]
    cv2.putText(prvaMala, str(kodaMala[1]), tuple(kodaMala[2]),
                                cv2.FONT_HERSHEY_SIMPLEX, fakPomanj, 255)


kode = kode[3:11]
kodeMale = kodeMale[3:11]
okolice = []
for koda in kode:
    xSr, ySr = koda[2][0], koda[2][1]
    okolice.append([prva[ySr-vOko:ySr+vOko, xSr-vOko:xSr+vOko], [xSr, ySr]])

vhodneHom = []
for okolica, koda, kodaMala in zip(okolice, kode, kodeMale):
    koda = izboljsajPolozajKode(koda, okolica)
    vhodneHom.append(koda[2])
    kodaMala[2] = [int(k*fakPomanj) for k in koda[2]]
#     cv2.circle(prvaMala, tuple(kodaMala[2]), 3, 255, 1, cv2.LINE_AA)
#     print("Natancen polozaj kode " + str(kodaMala[1]) + " je:")
#     print(koda[2])
# cv2.imshow('slika', prvaMala)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()

vhodneHom = np.float32(vhodneHom).reshape(-1,1,2)
izhodneHom = zapisiFizKoordinateKod(merilo, 2363.0, 500.0, 100.0, 100.0)

print(vhodneHom.shape)
print(izhodneHom.shape)


M, void = cv2.findHomography(vhodneHom, izhodneHom, cv2.LMEDS, 5.0)
# M = cv2.getPerspectiveTransform()

print(M)
# 
prvaMalaNova = cv2.warpPerspective(prvaMala, M, tuple(prvaMala.shape))

cv2.imshow('slika', prvaMalaNova)
cv2.waitKey(5000)
cv2.destroyAllWindows()

cv2.imwrite('prvaMalaNova.jpg', prvaMalaNova)







