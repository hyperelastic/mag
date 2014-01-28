

import numpy as np
import cv2
from cv2 import cv
import pickle




class SahovnicaPodatki:
    """Podatki sahovnice: Stevilo notranjih robov ter mera kvadratka."""
    
    def __init__(self, stNotrX, stNotrY, mera_mm=10):
        self.dimX = stNotrX
        self.dimY = stNotrY
        self.mera = mera_mm
    
    def vrniTupleDim(self):
        return (self.dimX, self.dimY)
    
    def vrniMero(self):
        return (self.mera)


class Kalibracija:
    """Kalibracija kamere s slikami sahovnice. Slike naj bodo oštevilčene od 0 dalje"""

    velikostSlike = ()
    velikostMale = ()
    velikostPoravnane = ()
    faktorPomanjsanja = 4
    faktorPovecanjaZaPoravnano = 1.0
    tmpUspeh = False
    tmpVogali = [[[]]]
    termKrit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1 )

    
    def __init__(self, sahovnicaPod):
        self.sahovnicaPodatki = sahovnicaPod
        self.uspesneItracije = []
        
    def vnesiSlike(self, potDoPrve, koncnica, stSlik):
        self.slike = []
        self.slikeMale = []
        for i in range(0, stSlik):
            print("Uvazam " + potDoPrve + str(i) + koncnica)
            self.slike.append(cv2.imread(potDoPrve + str(i) + koncnica, 
                                    cv2.CV_LOAD_IMAGE_GRAYSCALE))
            if i==0:
                Kalibracija.velikostSlike = np.shape(self.slike[0])[::-1]
                Kalibracija.velikostMale = tuple(x/Kalibracija.faktorPomanjsanja
                                                for x in Kalibracija.velikostSlike)
            
            self.slikeMale.append(cv2.resize(self.slike[-1],
                                        Kalibracija.velikostMale))

    def oceniVogalePomanjsanih(self):
        self.vogaliMali = []
        for i, slika in enumerate(self.slikeMale):
            print("Ocenjujem vogale slike " + str(i))
            Kalibracija.tmpUspeh, Kalibracija.tmpVogali = cv2.findChessboardCorners(slika, 
                                                            self.sahovnicaPodatki.vrniTupleDim())
            if Kalibracija.tmpUspeh:
                self.vogaliMali.append(Kalibracija.tmpVogali)
                self.uspesneItracije.append(i)
            else:
                print("Ocena vogalov slike " + str(i) + " ni bila uspesna")
    
    def prikaziOcenjeneVogale(self):
        for i in self.uspesneItracije:
            print("Uspesno so bile ocenjene koordinate vogalov slik: ", self.uspesneItracije)
            cv2.drawChessboardCorners(self.slikeMale[i], (8, 6), self.vogaliMali[i], True)
            cv2.imshow('Vogali ' + str(i), self.slikeMale[i])
            cv2.waitKey(0)
    
    def izboljsajVogale(self):
        self.vogali = [x*Kalibracija.faktorPomanjsanja for x in self.vogaliMali]
        
        for i in self.uspesneItracije:
            print("Izboljsujem koordinate vogalov slike " + str(i) + ".")
            cv2.cornerSubPix(self.slike[i], self.vogali[i], (10,10), (-1, -1),
                                Kalibracija.termKrit)
        
    def prikaziVogale(self):
        for i in self.uspesneItracije:
            print("Uspesno so bile ocenjene koordinate vogalov slik: ", self.uspesneItracije)
            cv2.drawChessboardCorners(self.slike[i], (8, 6), self.vogali[i], True)
            cv2.imshow('Vogali ' + str(i), self.slike[i])
            cv2.waitKey(2000)

    def poisciParametre(self):
        self.fizKoordinateTmp = []
        self.fizKoordinate = []
        for j in range(self.sahovnicaPodatki.dimY):
            for k in range(self.sahovnicaPodatki.dimX):
                self.fizKoordinateTmp.append([[k*self.sahovnicaPodatki.mera, 
                                                j*self.sahovnicaPodatki.mera, #0.0
                                                0.0]])
        for i in self.uspesneItracije:
            self.fizKoordinate.append(self.fizKoordinateTmp)
        
        self.fizKoordinate = np.array(self.fizKoordinate, dtype=np.float32)
        
        Kalibracija.tmpUspeh, self.matrikaKamere, self.popacitveniKoef, _, _=\
                                    cv2.calibrateCamera(self.fizKoordinate,
                                    self.vogali, Kalibracija.velikostSlike)

        print("\nMatrika kamere:")
        print(self.matrikaKamere)
        print("\nPopacitveni koeficienti:")
        print(self.popacitveniKoef)
        
    def poisciKartiPoravnanja(self):
        print("\nIscem karti poravnanja.")
        Kalibracija.velikostPoravnane = tuple(int(x*Kalibracija.faktorPovecanjaZaPoravnano)
                                                for x in Kalibracija.velikostSlike)
        self.matrikaKamereNova = np.array(self.matrikaKamere)
        self.karta1, self.karta2 = cv2.initUndistortRectifyMap(self.matrikaKamere,
                                                        self.popacitveniKoef, 
                                                        None,
                                                        self.matrikaKamereNova,
                                                        Kalibracija.velikostPoravnane,
                                                        cv2.CV_32FC1)

    def vloziKartiPoravnanja(self, imeDatoteke):
        print("\nVlagam karti poravnanja.")
        print(np.size(self.karta1))
        pickle.dump([self.karta1, self.karta2], open(imeDatoteke, "wb"))
        
        
        
        
class Poravnanje:
    """Poravnanje slike s pomočjo parametrov kalibracije."""
    
    def odvijKartiPoravnanja(self, imeDatoteke):
        self.karta1, self.karta2 = pickle.load( open( imeDatoteke, "rb" ) )
        
    def vnesiSliko(self, imeDatoteke):
        print("Uvazam sliko.")
        self.slika = cv2.imread(imeDatoteke)
        
    def poravnajSliko(self):
        self.poravnana = cv2.remap(self.slika, self.karta1, self.karta2, cv2.INTER_LINEAR)
    
    def pokaziPoravnano(self):
#         self.velikostPoravnane = np.shape(self.poravnana)[::-1]
#         self.velikostMale = tuple(x/4 for x in self.velikostPoravnane)
#         self.poravnana = cv2.resize(self.poravnana, self.velikostMale)
        cv2.imshow('Poravnana', self.poravnana)
        cv2.waitKey(0)
        
    def shraniPoravnano(self, imeDatoteke):
        cv2.imwrite(imeDatoteke, self.poravnana)
        


isciKalibracijo = False

if isciKalibracijo:
    sah = SahovnicaPodatki(8, 6, 27.0)
    kal = Kalibracija(sah)
    kal.vnesiSlike("slike/neobdelane/sahovnica/", ".jpg", 17)
    kal.oceniVogalePomanjsanih()
    # kal.prikaziOcenjeneVogale()
    kal.izboljsajVogale()
    # kal.prikaziVogale()
    kal.poisciParametre()
    kal.poisciKartiPoravnanja()
    kal.vloziKartiPoravnanja("kartiPoravnanja.p")



potDoSlik = "slike/neobdelane/linearniPoskus/preimenovane/"
potShranjevanje = "slike/neobdelane/linearniPoskus/odpacene/"

por = Poravnanje()
por.odvijKartiPoravnanja("kartiPoravnanja.p")

for i in range(0, 42):
    por.vnesiSliko(potDoSlik + str(i) + ".jpg")
    por.poravnajSliko()
    # por.pokaziPoravnano()
    por.shraniPoravnano(potShranjevanje + str(i) + ".jpg")






