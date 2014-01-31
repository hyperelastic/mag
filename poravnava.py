########### PORAVNAVA ###########
#   Zahteva paraParat/lokacijeKod in paraParat/faktorPomanjsanjaSlike
#   Ta skripta najprej poravna prvo sliko glede na kode z znanimi fiz. polozaji,






###IDEJA: KAJ PA, CE BI HOMOGRAFIJO OCENIL KAR IZ POVEZAV V OKOLICI KOD?###





import cv2
import numpy as np
from glob import glob

lokacijeKod = np.loadtxt("paraParat/lokacijeKod")
merilo = np.loadtxt("paraParat/faktorPomanjsanjaSlike")
fakPom = 0.2        #za grobo iskanje
vOko = 70           #velikost okolice kode za natancno iskanje



def izboljsajPolozajKode(koda, okolica):
    ''' Vrne kodo z izboljsanim polozajem '''
    
    orb = cv2.ORB(50, 2, 5)
    t1, op1 = orb.detectAndCompute(okolica[0],None)
    t2, op2 = orb.detectAndCompute(koda[0],None)
    naj = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ujemi = naj.match(op1,op2)
    ujemi = sorted(ujemi, key = lambda x:x.distance)

    notXY = np.array([ t1[d.queryIdx].pt for d in ujemi[:3] ])
    venXY = np.array([ t2[d.trainIdx].pt for d in ujemi[:3] ])
    premik = np.average(notXY-venXY, axis=0)
    stDevPremik = np.std(notXY-venXY, axis=0)
    
    delta = premik - (np.array(okolica[0].shape) - np.array(koda[0].shape))/2
    koda[2] = [x+d for x, d in zip(koda[2], delta)]
    
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
    
    return(koda)



imenaSlikKod = sorted(glob('slikeKoda/[0-20]*.jpg'))
kode = []
kodeM = []
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
for iSK in imenaSlikKod:
    koda = cv2.imread(iSK, cv2.IMREAD_GRAYSCALE)
    kode.append([koda, int(iSK[-6:-4]), [0, 0]])
    koda = cv2.resize(koda, (0,0), fx=fakPom, fy=fakPom, interpolation=cv2.INTER_AREA)
    kodeM.append([koda, int(iSK[-6:-4]), [0, 0]]) #poManjsane
    


imenaSlik = sorted(glob('slikeIzravnane/*.jpg'))
for iS in imenaSlik[:]:
    sl = cv2.imread(iS); print(iS)
    slC = cv2.imread(iS, cv2.IMREAD_GRAYSCALE)
    slC = clahe.apply(slC)
    slCM = cv2.resize(slC, (0,0), fx=fakPom, fy=fakPom, interpolation=cv2.INTER_AREA)
    

    for koda, kodaM in zip(kode, kodeM):
        res = cv2.matchTemplate(slCM, kodaM[0], cv2.TM_SQDIFF)
        void, void, zgLevo, void = cv2.minMaxLoc(res)
        sredinaM = [int(x/2+y) for (x, y) in zip(kodaM[0].shape, zgLevo)]
#         sredinaM = [x+y for x, y in zip(sredinaM, [-2, -3])]     # ZA TEST DELOVANJA IZBOLJSAVEPOLOZAJA
        kodaM[2] = [int(s) for s in sredinaM]
        koda[2] = [int(s/fakPom) for s in sredinaM]
        cv2.putText(slCM, str(kodaM[1]), tuple(kodaM[2]),
                                    cv2.FONT_HERSHEY_SIMPLEX, fakPom, 255)
    
    
    
    kodeUpo = kode[1:] #upostevane kode
    lokacijeKodUpo = lokacijeKod[1:] #ino ustrezne lokacije
    okolce = []
    for koda in kodeUpo:
        xSr, ySr = koda[2][0], koda[2][1]
        okolce.append([slC[ySr-vOko:ySr+vOko, xSr-vOko:xSr+vOko], [xSr, ySr]])



    vhodTke = []
    for o, koda in zip(okolce, kodeUpo):
        koda = izboljsajPolozajKode(koda, o)
        vhodTke.append(koda[2])
    
    
    
    vhodTke = np.float32(vhodTke).reshape(-1,1,2); print(vhodTke)
    izhodTke = np.float32(lokacijeKodUpo).reshape(-1,1,2); print(izhodTke)
    
    M, maska = cv2.findHomography(vhodTke, izhodTke, cv2.RANSAC, 5)
    
    print(M)
    print(maska.reshape(1,-1))
    
    sl2 = cv2.warpPerspective(sl, M, (int(np.max(izhodTke[:,:,0]))+180,
                                        int(np.max(izhodTke[:,:,1]))+180))
    
    cv2.imwrite("slikePoravnane/"+iS[-14:], sl2)
    
    
    
    
    
    
cv2.destroyAllWindows()
    





