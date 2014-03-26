import cv2
import numpy as np
from glob import glob
from operator import add
np.set_printoptions(precision=8)
np.set_printoptions(suppress=True) #onemogoci znanstveno

cv2.namedWindow("Prikaz", cv2.WINDOW_NORMAL)
imena = sorted(glob('./slikePoravnane/nelin/*.jpg'))

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

    
for ime in imena[465:466]:
    print(ime)
    sl = cv2.imread(ime)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6,6))
    sl = cv2.cvtColor(sl, cv2.COLOR_BGR2HSV)
    sl = cv2.insertChannel(clahe.apply(cv2.extractChannel(sl, 2)), sl, 2)
    sl = cv2.cvtColor(sl, cv2.COLOR_HSV2BGR)
    sl = cv2.medianBlur(sl, 7)
    izKan1 = [800, 2470, 1490, 3660] #samo obarvanost, izrez
    slKan1 = cv2.extractChannel(sl, 1)[izKan1[0]:izKan1[1],izKan1[2]:izKan1[3]]
    zacToc = [izKan1[1]-izKan1[0]-97, 100]

    kot=0
    kotPrej = 0
    radij = 20
    radijDiag = radij*2/np.sqrt(2)
    x = np.array(range(-radij, radij+1))
    for i in range(1):
        slOkence = 255-slKan1[zacToc[0]-radijDiag:zacToc[0]+radijDiag+1,
            zacToc[1]-radijDiag:zacToc[1]+radijDiag+1]
        if i==0: slOkence[:,:radijDiag] = np.fliplr(slOkence[:,-radijDiag:])

        #test s predpisano rotacijo
        M = cv2.getRotationMatrix2D((radijDiag,radijDiag), 19, 1)
        slOkence = cv2.warpAffine(slOkence, M, (int(2*radijDiag+1),int(2*radijDiag+1)))


        slOkence = cv2.GaussianBlur(slOkence, (7,7),9)
        gY, gX = np.gradient(np.float32(slOkence), 3, 3)
        slGx = np.uint8( 255 * (gX-np.min(gX)) / (np.max(gX)-np.min(gX) ))
        slGy = np.uint8( 255 * (gY-np.min(gY)) / (np.max(gY)-np.min(gY) ))
       
        cv2.imshow("Prikaz", slOkence); cv2.waitKey(10000)
        cv2.imshow("Prikaz", slGx); cv2.waitKey(10000)
        cv2.imshow("Prikaz", slGy); cv2.waitKey(10000)

        for i in range(15):
            gN = premakniSukaj(np.sin(kot)*gX + np.cos(kot)*gY,
                kot, 0, radij, radijDiag)
            slGn = np.uint8( 255 * (gN-np.min(gN)) / (np.max(gN)-np.min(gN) ))
            sila = np.sum(gN[radij,:])
            moment = np.dot(gN[radij,:], x)
            kotPrej = kot
            kot += 0.0001*moment - 0.1*(kot-kotPrej)
            print(np.array([kot, kot-kotPrej]))
            print(np.array([sila,moment]))
            cv2.imshow('Prikaz', slGn); cv2.waitKey(5000)
               
        
        #koti = np.linspace(0, 0.5*np.pi, num=19)
        #for j, kot in enumerate(koti):
        #    print(kot)
        #    gN = premakniSukaj(np.sin(kot)*gX + np.cos(kot)*gY,
        #        kot, 0, radij, radijDiag)
        #    slGn = np.uint8( 255 * (gN-np.min(gN)) / (np.max(gN)-np.min(gN) ))
        #    sila = np.sum(gN[radij,:])
        #    moment = np.dot(gN[radij,:], x)
        #    print(np.array([sila,moment]))
        #    cv2.imshow('Prikaz', slGn); cv2.waitKey(5000)





#slOkence = cv2.resize(slOkence, (0,0), fx=0.1, fy=0.1,
#    interpolation = cv2.INTER_AREA)



    #kot = 0.0
    #for j in range(6):
    #    print(j)
    #    slOkence = slKan1[zacToc[0]-radij:zacToc[0]+radij+1,
    #                         zacToc[1]-radij:zacToc[1]+radij+1]
    #    if j==0:
    #         slOkence[:,:radij] = np.fliplr(slOkence[:,-radij:])
    #         #slOkence[-radij:,:] = np.flipud(slOkence[:radij,:])
    #    #pas = np.mean(np.float64(slOkence), axis=1)[radij-5:radij+6]
    #    #poliKonst = np.polyfit(range(-5, 6), pas, 2)
    #    #lokMax = -poliKonst[1]/2/poliKonst[0]
    #    #print(lokMax)

    #    #M = cv2.getRotationMatrix2D((2*radij,2*radij), 5, 1)
    #    #slOkence = cv2.warpAffine(slOkence, M, np.shape(slOkence))
    #    cv2.imshow("Prikaz", slOkence); cv2.waitKey(50000)
    #    

    #    #inkrementalno iskanje kota najmanjse svetlosti
    #    #alfe = np.linspace(kot-0.1*np.pi, kot+0.1*np.pi, num=29)
    #    alfe = np.linspace(-0.2*np.pi, +0.2*np.pi, num=9)

    #    metrike = []
    #    for alfa in alfe: 
    #        M = cv2.getRotationMatrix2D((radij,radij), np.degrees(kot+alfa), 1)
    #        pas = cv2.warpAffine(slOkence, M, (2*radij, 2*radij))[radij-2:radij+3,:]
    #        metrike.append(np.sum(pas))
    #        cv2.imshow("Prikaz", pas)
    #        cv2.waitKey(50)
    #    metrike = np.array(metrike)
    #    poliKonst = np.polyfit(alfe, np.array(metrike), 2)
    #    print(np.degrees(alfe)); print(metrike); 

    #    kot = -poliKonst[1]/2/poliKonst[0]
    #    print(np.degrees(kot))
    #    
    #    #M = cv2.getRotationMatrix2D((radij,radij), np.degrees(kot), 2/np.sqrt(2))
    #    #slOkence = cv2.warpAffine(slOkence, M, np.shape(slOkence))
    #    #pas = np.mean(np.float64(slOkence), axis=1)
    #    #poliKonst = np.polyfit(range(np.argmin(pas)-5, np.argmin(pas)+6), 
    #    #                        pas[np.argmin(pas)-5:np.argmin(pas)+6], 2)
    #    #delta = -poliKonst[1]/2/poliKonst[0]-radij
    #    #print(delta); print(zacToc)
    #    #zacToc = map(add, zacToc, [delta*np.cos(kot), -delta*np.sin(kot)])
    #    
    #    
    #    print(zacToc)
    #    zacToc = map(add, zacToc, [radij*np.sin(kot), radij*np.cos(kot)])
    #    print(zacToc)
    #    
    #    cv2.circle(slKan1, (np.int(zacToc[1]),np.int(zacToc[0])), 1, 0, thickness=1)

        #cv2.imshow("Prikaz", slOkence); cv2.waitKey(10000)


        #slOkence = np.fliplr(slOkence)
        #slOkence = cv2.linearPolar(slOkence, (2*radij, 2*radij), int(1.0*radij),
        #                            cv2.INTER_LINEAR)
        #slOkence = cv2.GaussianBlur(slOkence, (9, 3), 6)
        #slPas = slOkence[:,-1:]

        ###aproksimacija s kubicnim polinomom za maksimum
        #konstantePolinom = np.polyfit(np.array(range(-4, 5)),
        #                                slPas[lokMax-4:lokMax+5], 2)
        #print(lokMax); print("lokMax")
        #lokMax += -konstantePolinom[1][0]/2/konstantePolinom[0][0]
        #print(lokMax)
        #kot = 0.5*np.pi*((lokMax-2*radij)/radij)
        #print(180/np.pi*kot)
        #zacToc = map(add, zacToc, [-radij*np.sin(kot), radij*np.cos(kot)])
        #cv2.imshow("Prikaz", slOkence); cv2.waitKey(10000)
        ##cv2.imshow("Prikaz", slPas); cv2.waitKey(10000)
        
        
#        #slOkence = cv2.GaussianBlur(slOkence, (1, 1), 2)
#
#        slPas = slOkence[1.2*radij:-0.5*radij,-0.2*radij-1:]
#        slTemplate = 180*np.uint8(np.ones((9,np.shape(slPas)[1])))
#        slTemplate[2:7,:] = np.uint8(40+np.zeros((5,np.shape(slPas)[1])))
#        #slTemplate = cv2.GaussianBlur(slTemplate, (3, 3), 1.0)
#
#        #cv2.imshow("Prikaz", slPas)
#        #cv2.waitKey(1000)
#        #cv2.imshow("Prikaz", slTemplate)
#        #cv2.waitKey(1000)
#
#        #priblizno iskanje maksimuma
#        slPasMatch = cv2.matchTemplate(slPas, slTemplate, cv2.TM_CCORR_NORMED)
#        void, void, void, maxLoc = cv2.minMaxLoc(slPasMatch); 
#        maxLoc = maxLoc[1] + 4.5 #zamik v velikosti slTemplate/2
#        print(maxLoc)
#
#        #aproksimacija s kubicnim polinomom za maksimum
#        slPas = 255-np.uint8(np.mean(slPas, axis=1))
#        slPas = cv2.GaussianBlur(slPas, (3, 3), 3)
#        konstantePolinom = np.polyfit(np.array(range(-4, 5)),
#                                        slPas[maxLoc-4:maxLoc+5], 2)
#        maxLoc += -konstantePolinom[1][0]/2/konstantePolinom[0][0]
#        print(maxLoc)
#        maxLoc += 1.2*radij    #postavitev nazaj v koordinate slOkenca
#        print(maxLoc)
#        #cv2.imshow("Prikaz", slPas)
#        #cv2.waitKey(20000)
#        #kot = 0.05*(19*kot + 0.5*np.pi*(float(maxLoc-2*radij)/radij)) #vztrajnost
#        #kot = 0.1*(9*kot + 0.5*np.pi*(float(maxLoc-2*radij)/radij)) #vztrajnost
#        kot =  0.5*np.pi*((maxLoc-2*radij)/radij)      #brez vztr.
#        zacToc = [zacToc[0]-radij*np.sin(kot), zacToc[1]+radij*np.cos(kot)]
#        print("kot: " + str(kot) + "\n")
#
#        #cv2.imshow("Prikaz", slPasMatch)
#        #cv2.waitKey(10000)
#
#        cv2.circle(slOkence, (3*radij, int(maxLoc)), 1, 255, thickness=1)
#        cv2.circle(slKan1, tuple([int(zacToc[1]), 
#                            int(zacToc[0])]), 2, 215, thickness=2)
#        #cv2.imshow("Prikaz", slOkence)
#        #cv2.waitKey(50)
#    #cv2.imshow("Prikaz", slKan1[800:,:-1200])
#    #cv2.imshow("Prikaz", slKan1[800:,-1000:])
#    #cv2.imshow("Prikaz", slKan1)
#    cv2.imshow("Prikaz", slKan1[:,:-1100])
#    cv2.waitKey(100000)

#kanal1 = cv2.extractChannel(sl, 1)
#visIzreza = 800; sirIzreza = 1500
#kanal1 = kanal1[visIzreza:-50, sirIzreza:-160]
#clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(102,102))
#kanal1 = clahe.apply(kanal1)

#pas = kanal1[1550:-70, 120:130]
#pas = cv2.GaussianBlur(pas, (51,21), 0)
#void, void, lMaxKanal1, void = cv2.minMaxLoc(pas)
#lMaxKanal1 = [lMaxKanal1[0]+120, lMaxKanal1[1]+1550]
