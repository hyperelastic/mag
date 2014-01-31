########### PORAVNAVA PRVE SLIKE ###########
#   Kode morajo bit lepo centrirane.
#   Prva slika naj bo taka, da so lepo vidne vse kode.
#   Ta skripta najprej poravna prvo sliko glede na kode z znanimi fiz. polozaji,
#nato s homografijo oceni se fiz. polozaje drugih kod in vse skupaj shrani v
#besedilno datoteko paraParat/lokacijeKod. Poleg tega shrani parameter fakPomanj,
#ki doloca merilo v datoteko "paraParat/faktorPomanjsanjaSlike".

import cv2
import numpy as np
from glob import glob



imePrve = sorted(glob('slikeIzravnane/*.jpg'))[7]
fakPomanj = 0.4     #faktor pomanjsanja slike za grobo iskanje
merilo = 1.5        #merilo slikovneDimenzije/fizikalneDimenzije
vOko = 70           #velikost okolice kode za natancno iskanje



def izboljsajPolozajKode(koda, okolica):
    ''' Vrne kodo z izboljsanim polozajem '''
    
    orb = cv2.ORB(50, 2, 5)
    t1, op1 = orb.detectAndCompute(okolica[0],None)
    t2, op2 = orb.detectAndCompute(koda[0],None)
    naj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ujemi = naj.match(op1,op2)
    ujemi = sorted(ujemi, key = lambda x:x.distance)

    notXY = np.array([ t1[d.queryIdx].pt for d in ujemi[:4] ])
    venXY = np.array([ t2[d.trainIdx].pt for d in ujemi[:4] ])
    premik = np.average(notXY-venXY, axis=0)
    stDevPremik = np.std(notXY-venXY, axis=0)
    
    delta = premik - (np.array(okolica[0].shape) - np.array(koda[0].shape))/2
    koda[2] = [x+d for x, d in zip(koda[2], delta)]
    
    print("premik"); print(premik)
    print("stDevPremik"); print(stDevPremik)
    print("delta"); print(delta)
    print("\n")
    srKodeNaKodi = [x/2 for x in koda[0].shape]
    srKodeNaOkolici = tuple([sr+d for sr, d in zip(srKodeNaKodi, premik)])
    cv2.circle(okolica[0], (int(srKodeNaOkolici[0]),
                                int(srKodeNaOkolici[1])),
                                    8, 255, 1, cv2.LINE_AA)

    sl3 = cv2.drawMatches(okolica[0], t1, koda[0], t2, ujemi[:4], 200, flags=2)
    cv2.imshow("Izboljsava", sl3)
    cv2.waitKey(1500)
    
    return(koda)



def fizKoordinateVogalov(merilo, zamikX, zamikY):
    Tocke = np.array([  [0.0, 1500.0],
                        [2361.0, 1500.0],
                        [2365.0, 0.0],
                        [0.0, 0.0]   ])

    Tocke[:,1] = Tocke[:,1]+zamikX
    Tocke[:,0] = Tocke[:,0]+zamikY
    return(np.float32(Tocke*merilo).reshape(-1,1,2))


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
prva = cv2.imread(imePrve, cv2.IMREAD_GRAYSCALE)
prva = clahe.apply(prva)
prvaMala = cv2.resize(prva, (0,0), fx=fakPomanj, fy=fakPomanj, interpolation=cv2.INTER_AREA)



kode = []
kodeMale = []
imenaSlikKod = sorted(glob('slikeKoda/[0-20]*.jpg'))
for iSK in imenaSlikKod:
    koda = cv2.imread(iSK, cv2.IMREAD_GRAYSCALE)
    kode.append([koda, int(iSK[-6:-4]), [0, 0]])
    koda = cv2.resize(koda, (0,0), fx=fakPomanj, fy=fakPomanj, interpolation=cv2.INTER_AREA)
    kodeMale.append([koda, int(iSK[-6:-4]), [0, 0]])



for koda, kodaMala in zip(kode, kodeMale):
    res = cv2.matchTemplate(prvaMala, kodaMala[0], cv2.TM_SQDIFF)
    void, void, zgorajLevo, void = cv2.minMaxLoc(res)
    sredinaMala = [int(x/2+y) for (x, y) in zip(kodaMala[0].shape, zgorajLevo)]
#     sredinaMala = [x+y for x, y in zip(sredinaMala, [4, 3])]     # ZA TEST DELOVANJA IZBOLJSAVEPOLOZAJA
    kodaMala[2] = [int(s) for s in sredinaMala]
    koda[2] = [int(s/fakPomanj) for s in sredinaMala]



kodeRobovi = [kode[i] for i in [3,4,7,8]]
kodeRoboviMale = [kodeMale[i] for i in [3,4,7,8]]



okolice = []
for koda in kodeRobovi:
    xSr, ySr = koda[2][0], koda[2][1]
    okolice.append([prva[ySr-vOko:ySr+vOko, xSr-vOko:xSr+vOko], [xSr, ySr]])



vhodniVogali = []
for okolica, koda in zip(okolice, kodeRobovi):
    koda = izboljsajPolozajKode(koda, okolica)
    cv2.destroyAllWindows()
    vhodniVogali.append(koda[2])



for koda in kode:
    cv2.putText(prva, str(koda[1]), tuple([int(x+30) for x in koda[2]]),
                                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.8, 255, 2)



vhodniVogali = np.float32(vhodniVogali).reshape(-1,1,2)
izhodniVogali = fizKoordinateVogalov(merilo, 120.0, 120.0)
M = cv2.getPerspectiveTransform(vhodniVogali, izhodniVogali)



vhodneKoordinateVseh = []
for koda in kode:
    vhodneKoordinateVseh.append(koda[2])
vhodneKoordinateVseh = np.float32(vhodneKoordinateVseh).reshape(-1,1,2)
fizKoordinateVseh = cv2.perspectiveTransform(vhodneKoordinateVseh, M)
print(vhodneKoordinateVseh)
print(fizKoordinateVseh)
print fizKoordinateVseh[:,0,:]



prvaNova = cv2.warpPerspective(prva, M, (int(3698+120*merilo), int(2400+120*merilo)))
# for fk in fizKoordinateVseh[:, 0, :]:
#     cv2.circle(prvaNova, tuple(fk), 8, 255, 1, cv2.LINE_AA)



cv2.imwrite("prvaNova.jpg", prvaNova)


np.savetxt("paraParat/lokacijeKod", fizKoordinateVseh[:,0,:])
np.savetxt("paraParat/faktorPomanjsanjaSlike", [fakPomanj])















