#!/usr/bin/python2
import cv2
import numpy as np
from glob import glob
from operator import add
#from io import BytesIO #SAMO ZA TEST
#datoteka = BytesIO() #SAMO ZA TEST
import matplotlib.pyplot as plt
np.set_printoptions(precision=1)
np.set_printoptions(suppress=True) #onemogoci znanstveno

imeTabele = './tabele/nelin.txt'
imena = sorted(glob('./slikePoravnane/nelin/*.jpg'))
stNiza = 7
nizi = [[27, 67], [97, 125],
        [149, 181], [198, 232],
        [246, 285], [296, 334],
        [347, 385], [395, 426]]
imenaSlikKod = sorted(glob('slikeKoda/[0-20]*.jpg'))
vsiPodatki = []
cv2.namedWindow("Prikaz", cv2.WINDOW_NORMAL)

def premakniSukaj(slika, kot, delta, radij, radijDiag):
    """ Vrne sliko velikosti (2radij+1,2radij+1).

    slika: slika velikosti (2*radijDiag+1, 2*radijDiag+1)
    kot: kot v radianih
    delta: velikost navpicnega premika po rotaciji
    radij: premer vcrtanega kroga koncne
    radijDiag: premer vcrt. kroga zacetne
    """
    M = cv2.getRotationMatrix2D((radijDiag,radijDiag), np.degrees(kot), 1)
    M[1,-1] += delta-(radijDiag-radij)
    M[0,-1] += -(radijDiag-radij)
    return(cv2.warpAffine(slika, M, (2*radij+1, 2*radij+1)))

for ime in imena[nizi[stNiza][0]:nizi[stNiza][1]]:
    #zacetna obdelava slike
    print(ime)
    sl = cv2.imread(ime)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6,6))
    sl = cv2.cvtColor(sl, cv2.COLOR_BGR2HSV)
    sl = cv2.insertChannel(clahe.apply(cv2.extractChannel(sl, 2)), sl, 2)
    sl = cv2.cvtColor(sl, cv2.COLOR_HSV2BGR)
    sl = cv2.medianBlur(sl, 7)

    #nastavitve iskanja tock
    radij = 40
    korak = 10
    izKan1 = [800, 2470, 1490, 3660] #samo obarvanost, izrez
    slKan1 = cv2.extractChannel(sl, 1)[izKan1[0]:izKan1[1],izKan1[2]:izKan1[3]]
    zacToc = [izKan1[1]-izKan1[0]-97, 100]
    radijDiag = radij*2/np.sqrt(2)
    delta=0; deltaPrej=0
    kot=0; kotPrej = 0
    x = np.array(range(-radij, radij+1)) #za izracun momenta
    tocke = []

    #zanka iskanje tock
    stIteracij = int(np.round(1945/korak, 0))
    for i in range(stIteracij):
        slOkence = slKan1[zacToc[0]-radijDiag:zacToc[0]+radijDiag+1,
            zacToc[1]-radijDiag:zacToc[1]+radijDiag+1]
        if i==0: slOkence[:,:radijDiag] = np.fliplr(slOkence[:,-radijDiag:])
        zacToc = np.float64(zacToc)
        ##test s predpisano rotacijo
        #M = cv2.getRotationMatrix2D((radijDiag,radijDiag), -35, 1)
        #slOkence = cv2.warpAffine(slOkence, M, (int(2*radijDiag+1),int(2*radijDiag+1)))
        slOkence = cv2.GaussianBlur(slOkence, (3,3),3)
        gY, gX = np.gradient(np.float32(slOkence), 3, 3)
        ##prikaz gradientov in okenca
        #cv2.imshow("Prikaz", slOkence); cv2.waitKey(500)
        #slGx = np.uint8( 255 * (gX-np.min(gX)) / (np.max(gX)-np.min(gX) ))
        #slGy = np.uint8( 255 * (gY-np.min(gY)) / (np.max(gY)-np.min(gY) ))
        #cv2.imshow("Prikaz", slGx); cv2.waitKey(10000)
        #cv2.imshow("Prikaz", slGy); cv2.waitKey(10000)
        ##test vektorske vsote gradientov
        #koti = np.linspace(0, 1.0*np.pi, num=150)
        #for kot in koti:
        #    print(np.degrees(kot))
        #    gN = np.sin(kot)*gX + np.cos(kot)*gY
        #    gN = premakniSukaj(gN, -kot, 0, radij, radijDiag)
        #    slGn = np.uint8( 255 * (gN-np.min(gN)) / (np.max(gN)-np.min(gN) ))
        #    cv2.imshow('Prikaz', slGn); cv2.waitKey(50)
        delta = deltaPrej = 0
        for void in range(3):
            for j in range(25): #odmicno uravnotezenje
                gN = premakniSukaj(np.sin(kot)*gX + np.cos(kot)*gY,
                    -kot, delta, radij, radijDiag)
                sila = np.sum(gN[radij,:])
                delta += 0.02/radij*sila; delta -= 0.1*(delta-deltaPrej)
                deltaPrej = delta; kotPrej = kot
                ##prikaz poteka
                #print(np.array([sila, delta]))
                #slGn = np.uint8( 255 * (gN-np.min(gN)) / (np.max(gN)-np.min(gN) ))
                #cv2.imshow('Prikaz', slGn); cv2.waitKey(10)
            #slGn = np.uint8( 255 * (gN-np.min(gN)) / (np.max(gN)-np.min(gN) ))
            #cv2.imshow('Prikaz', slGn); cv2.waitKey(10)
            #print(delta)
            for j in range(35): #kotno uravnotezenje
                gN = premakniSukaj(np.sin(kot)*gX + np.cos(kot)*gY,
                    -kot, delta, radij, radijDiag)
                moment = np.dot(gN[radij,:], x)
                kot += 0.0001/radij*moment; kot -= 0.1*(kot-kotPrej)
                ##prikaz poteka
                #print(np.array([moment, kot]))
                #slGn = np.uint8( 255 * (gN-np.min(gN)) / (np.max(gN)-np.min(gN) ))
                #cv2.imshow('Prikaz', slGn); cv2.waitKey(10)
            #slGn = np.uint8( 255 * (gN-np.min(gN)) / (np.max(gN)-np.min(gN) ))
            #cv2.imshow('Prikaz', slGn); cv2.waitKey(10)
            #print(np.degrees(kot))
        zacToc = map(add, zacToc, [-delta*np.cos(kot), -delta*np.sin(kot)])
        if zacToc[0] < izKan1[1]-izKan1[0]:
            tocke.append(zacToc)
            zacToc = map(add, zacToc, [-korak*np.sin(kot), korak*np.cos(kot)])
        else:
            print("Nosilec gre prevec navzdol, prekinjam zajem za to sliko\
                    \nTocke do konca bodo zapolnjene z [-1.0, -1.0]")
            tocke += (stIteracij-i)*[[10000.0,10000.0]]
            break

    #prikaz in shranjevanje slike
    #for t in tocke:
    #    cv2.circle(slKan1, tuple([int(np.round(a)) for a in t][::-1]),
    #                1, 255, thickness=2)
    #cv2.imshow('Prikaz', slKan1); cv2.waitKey(10)
    #cv2.imwrite("./slikePikice/" + imeTabele[9:14] + "Korak" + str(korak) +\
    #    "-" + ime[-14:], slKan1)

    #iskanje dolzine vzmeti
    koda = cv2.imread(imenaSlikKod[0], cv2.IMREAD_GRAYSCALE)
    izSl = [300, 2200, 600, 3600]
    sl = cv2.cvtColor(sl, cv2.COLOR_BGR2GRAY)[izSl[0]:izSl[1],izSl[2]:izSl[3]] 
    slika = cv2.matchTemplate(sl, koda, cv2.TM_SQDIFF)
    void, void, lokKode, void = cv2.minMaxLoc(slika)
    lokKode = [lK+int(float(sK)/2) for lK, sK in zip(lokKode, np.shape(koda))][::-1]
    if tocke[-1][0] < 9999:
        lokKonca = [t+izK-izS for t,izK,izS in zip(tocke[-1], izKan1[::2], izSl[::2])]
        razdalja = np.array([lDe-lCa for lCa, lDe in zip(lokKonca, lokKode)])
    else:
        print("Ne morem najti lokacije konca nosilca, ker ga ni v sliki.")
        razdalja = [10000, 10000]
    razdalja = np.reshape(razdalja, (2,1))
    print("Vnasam razdaljo [%(r1)0.2f, %(r2)0.2f]" % \
            {"r1":razdalja[0], "r2":razdalja[1]})

    #priprava na shranjevanje tabele
    cas = np.transpose(np.array([2*[int(ime[-10:-4])]]))
    podatki = np.hstack((razdalja, cas, np.transpose(np.array(tocke))))
    if len(vsiPodatki) is 0: vsiPodatki = podatki
    else: vsiPodatki = np.vstack((vsiPodatki, podatki))

    ##prikaz tock na sliki
    #lokKonca = tuple([int(l) for l in lokKonca])
    #lokKode = tuple(lokKode)
    #cv2.circle(sl, lokKode[::-1], 10, 255, thickness=4) 
    #cv2.circle(sl, lokKonca[::-1], 10, 255, thickness=4) 
    #cv2.imshow("Prikaz", koda); cv2.waitKey(10000)
    #cv2.imshow("Prikaz", sl[:,:]); cv2.waitKey(10000)

#shranjevanje tabele
np.savetxt(imeTabele[:-4] + "Niz" + str(stNiza) +
    "Korak" + str(korak) + imeTabele[-4:], vsiPodatki,
    fmt=tuple(["%.2f", "%i"] + ["%.2f" for i in\
        range(np.shape(vsiPodatki)[1]-2)]),
    delimiter=" ",
    header= "\nDokument z dolzinami vzmeti, casi zajemov in tockami.\n\n" +
            "Obdelane so slike v obsegu od '" + imena[0] +
            "' do '" + imena[-1] + "'.\n\n" +
            "Podatki so shranjeni v sledeci obliki:\n" +
            "DolzinaVzmetiY[0], Cas[0], TockeY[0]\n" +
            "DolzinaVzmetiX[0], Cas[0], TockeX[0]\n" +
            "DolzinaVzmetiY[1], Cas[1], TockeY[1]\n" +
            "DolzinaVzmetiX[1], Cas[1], TockeX[1]\n" +
            "\t...\n"+
            "DolzinaVzmetiY[:], Cas[:], TockeY[:]\n" +
            "DolzinaVzmetiX[:], Cas[:], TockeX[:]\n\n" + 
            "Koordinate x narascajo desno, y pa navzdol.\n" +
            "Ce je 'DolzinaVzmeti' 10000.0, je bil nosilec povesen " +
            "preko meja slike in ni bilo mogoce zajeti vseh tock.\n" +
            "Nezajete tocke imajo koordinate 10000.0.\n\n"
            )
#print datoteka.getvalue()

##graf s tockami
#    tocke = np.array(tocke)
#    plt.plot(tocke[:,1], -tocke[:,0], 'r.-')
#plt.axis('equal')
#plt.show()

cv2.destroyAllWindows()





