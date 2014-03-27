import cv2
import numpy as np
from glob import glob
from operator import add
import matplotlib.pyplot as plt
np.set_printoptions(precision=5)
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

    
for ime in imena[437:465]:
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

    radij = 60
    korak = 30
    radijDiag = radij*2/np.sqrt(2)
    delta=0; deltaPrej=0
    kot=0; kotPrej = 0
    x = np.array(range(-radij, radij+1)) #za izracun momenta
    tocke = []
    for i in range(int(np.round(1960/korak, 0))):
        slOkence = slKan1[zacToc[0]-radijDiag:zacToc[0]+radijDiag+1,
            zacToc[1]-radijDiag:zacToc[1]+radijDiag+1]
        if i==0: slOkence[:,:radijDiag] = np.fliplr(slOkence[:,-radijDiag:])
        zacToc = np.float64(zacToc)

        ##test s predpisano rotacijo
        #M = cv2.getRotationMatrix2D((radijDiag,radijDiag), -35, 1)
        #slOkence = cv2.warpAffine(slOkence, M, (int(2*radijDiag+1),int(2*radijDiag+1)))

        slOkence = cv2.GaussianBlur(slOkence, (3,3),3)
        #cv2.imshow("Prikaz", slOkence); cv2.waitKey(500)
        gY, gX = np.gradient(np.float32(slOkence), 3, 3)

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
            for i in range(25): #odmicno uravnotezenje
                gN = premakniSukaj(np.sin(kot)*gX + np.cos(kot)*gY,
                    -kot, delta, radij, radijDiag)
                sila = np.sum(gN[radij,:])
                delta += 0.02/radij*sila; delta -= 0.1*(delta-deltaPrej)
                deltaPrej = delta; kotPrej = kot
                #print(np.array([sila, delta]))
                #slGn = np.uint8( 255 * (gN-np.min(gN)) / (np.max(gN)-np.min(gN) ))
                #cv2.imshow('Prikaz', slGn); cv2.waitKey(10)
            #slGn = np.uint8( 255 * (gN-np.min(gN)) / (np.max(gN)-np.min(gN) ))
            #cv2.imshow('Prikaz', slGn); cv2.waitKey(10)
            #print(delta)
            for i in range(35): #kotno uravnotezenje
                gN = premakniSukaj(np.sin(kot)*gX + np.cos(kot)*gY,
                    -kot, delta, radij, radijDiag)
                moment = np.dot(gN[radij,:], x)
                kot += 0.0001/radij*moment; kot -= 0.1*(kot-kotPrej)
                #print(np.array([moment, kot]))
                #slGn = np.uint8( 255 * (gN-np.min(gN)) / (np.max(gN)-np.min(gN) ))
                #cv2.imshow('Prikaz', slGn); cv2.waitKey(10)
            #slGn = np.uint8( 255 * (gN-np.min(gN)) / (np.max(gN)-np.min(gN) ))
            #cv2.imshow('Prikaz', slGn); cv2.waitKey(10)
            print(np.degrees(kot))
        zacToc = map(add, zacToc, [-delta*np.cos(kot), -delta*np.sin(kot)])
        tocke.append(zacToc)
        zacToc = map(add, zacToc, [-korak*np.sin(kot), korak*np.cos(kot)])

    for t in tocke:
        cv2.circle(slKan1, (int(np.round(t[1], 0)), int(np.round(t[0], 0))),
            1, 255, thickness=1) 

    #cv2.imshow('Prikaz', slKan1[100:,:-500]); cv2.waitKey(3000)
    cv2.imwrite("./slikePikice/" + ime[-14:], slKan1)

    tocke = np.array(tocke)
    plt.plot(tocke[:,1], -tocke[:,0], 'r.-')
plt.axis('equal')
plt.show()
       





