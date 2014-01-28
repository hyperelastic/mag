
import numpy as np
import cv2
import glob

# pogoji za prekinitev
pogoji = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
velKvadr = 27.0 
pomanj = 0.12

# priprava tock predmetov, npr (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
Tprip = np.zeros((6*8,3), np.float32)
Tprip[:,:2] = velKvadr*np.mgrid[0:8,0:6].T.reshape(-1,2)

print(Tprip)

# Vektorja za hranjenje tock v prostoru in na ravnini slike iz vseh slik
T3d = [] # 3d tocke v prostoru resnicnega sveta
T2d = [] # 2d tocke v ravnini slike.

slike = glob.glob('slikeSahovnica/*.jpg')
for s in slike[:]:
    print("Obdelujem sliko " + str(s))
    slika = cv2.imread(s, cv2.IMREAD_GRAYSCALE)
    mala = cv2.resize(slika, (0,0), fx=pomanj, fy=pomanj, interpolation=cv2.INTER_AREA)
    uspeh, vogaliMali = cv2.findChessboardCorners(mala, (8,6),None)

    # Izboljsa in doda 3d in 2d tocke, ce jih najde.
    if uspeh == True:
        print("Sahovnica najdena. Izboljsujem in iscem vogale.\n")
        T3d.append(Tprip)
        vogali = cv2.cornerSubPix(slika,vogaliMali/pomanj,(11,11),(-1,-1),pogoji)
        T2d.append(vogali)

        # Narise in prikaze vogale
        mala = cv2.drawChessboardCorners(mala, (8,6), vogali*pomanj, uspeh)
        cv2.imshow('slika',mala)
        cv2.waitKey(100)
cv2.destroyAllWindows()

uspeh, matr, popac, rvecs, tvecs = cv2.calibrateCamera(T3d, T2d, slika.shape[::-1],None,None)
if uspeh:
    print("Shranjujem matriko in popacenje fotoaparata.")
    np.savetxt("paraParat/matrikaFotoaparata", matr)
    np.savetxt("paraParat/popacenjeFotoaparata", popac)





