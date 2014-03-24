import cv2
import numpy as np
from glob import glob
from operator import add

cv2.namedWindow("Prikaz", cv2.WINDOW_NORMAL)
imena = sorted(glob('./slikePoravnane/nelin/*.jpg'))

    
kot = 0
for ime in imena[155:156]:
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
    cv2.circle(slKan1, (zacToc[1], zacToc[0]), 1, 255, thickness=1)

    #cv2.imshow("Prikaz", slKan1[1300:,:-1900])
    #cv2.waitKey(2000)
    
    kot = 0
    radij = 10
    for j in range(194):
        print(j)
        slOkence = slKan1[zacToc[0]-2*radij:zacToc[0]+2*radij,
                             zacToc[1]-2*radij:zacToc[1]+2*radij]
        slOkence = np.fliplr(slOkence)
        slOkence = cv2.linearPolar(slOkence, (2*radij, 2*radij), int(1.1*radij),
                                    cv2.INTER_LINEAR)
        #slOkence = cv2.GaussianBlur(slOkence, (1, 1), 2)

        slPas = slOkence[1.2*radij:-0.5*radij,-0.2*radij-1:]
        slTemplate = 180*np.uint8(np.ones((9,np.shape(slPas)[1])))
        slTemplate[2:7,:] = np.uint8(40+np.zeros((5,np.shape(slPas)[1])))
        #slTemplate = cv2.GaussianBlur(slTemplate, (3, 3), 1.0)

        #cv2.imshow("Prikaz", slPas)
        #cv2.waitKey(1000)
        #cv2.imshow("Prikaz", slTemplate)
        #cv2.waitKey(1000)

        #priblizno iskanje maksimuma
        slPasMatch = cv2.matchTemplate(slPas, slTemplate, cv2.TM_CCORR_NORMED)
        void, void, void, maxLoc = cv2.minMaxLoc(slPasMatch); 
        maxLoc = maxLoc[1] + 4.5 #zamik v velikosti slTemplate/2
        print(maxLoc)

        #aproksimacija s kubicnim polinomom za maksimum
        slPas = 255-np.uint8(np.mean(slPas, axis=1))
        slPas = cv2.GaussianBlur(slPas, (3, 3), 3)
        konstantePolinom = np.polyfit(np.array(range(-4, 5)),
                                        slPas[maxLoc-4:maxLoc+5], 2)
        maxLoc += -konstantePolinom[1][0]/2/konstantePolinom[0][0]
        print(maxLoc)
        maxLoc += 1.2*radij    #postavitev nazaj v koordinate slOkenca
        print(maxLoc)
        #cv2.imshow("Prikaz", slPas)
        #cv2.waitKey(20000)
        

        #kot = 0.05*(19*kot + 0.5*np.pi*(float(maxLoc-2*radij)/radij)) #vztrajnost
        #kot = 0.1*(9*kot + 0.5*np.pi*(float(maxLoc-2*radij)/radij)) #vztrajnost
        kot =  0.5*np.pi*((maxLoc-2*radij)/radij)      #brez vztr.
        zacToc = [zacToc[0]-radij*np.sin(kot), zacToc[1]+radij*np.cos(kot)]
        print("kot: " + str(kot) + "\n")

        #cv2.imshow("Prikaz", slPasMatch)
        #cv2.waitKey(10000)

        cv2.circle(slOkence, (3*radij, int(maxLoc)), 1, 255, thickness=1)
        cv2.circle(slKan1, tuple([int(zacToc[1]), 
                            int(zacToc[0])]), 2, 215, thickness=2)
        #cv2.imshow("Prikaz", slOkence)
        #cv2.waitKey(50)
    #cv2.imshow("Prikaz", slKan1[800:,:-1200])
    cv2.imshow("Prikaz", slKan1[800:,-1000:])
    #cv2.imshow("Prikaz", slKan1)
    cv2.waitKey(100000)

#kanal1 = cv2.extractChannel(sl, 1)
#visIzreza = 800; sirIzreza = 1500
#kanal1 = kanal1[visIzreza:-50, sirIzreza:-160]
#clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(102,102))
#kanal1 = clahe.apply(kanal1)

#pas = kanal1[1550:-70, 120:130]
#pas = cv2.GaussianBlur(pas, (51,21), 0)
#void, void, lMaxKanal1, void = cv2.minMaxLoc(pas)
#lMaxKanal1 = [lMaxKanal1[0]+120, lMaxKanal1[1]+1550]

#cv2.namedWindow("Prikaz", cv2.WINDOW_NORMAL)
#kot = 0
#for i in range(195):
#    print(i)
#    radij = 10
#    okence = kanal1[lMaxKanal1[1]-2*radij:lMaxKanal1[1]+2*radij+1,
#                        lMaxKanal1[0]-2*radij:lMaxKanal1[0]+2*radij+1]
#    okence = np.fliplr(okence)
#    okence = cv2.linearPolar(okence, (2*radij, 2*radij), int(1.1*radij),
#                                cv2.INTER_LINEAR)
#    #pas = okence[(1+kot/0.5/np.pi)*radij:\
#    #                        (3+kot/0.5/np.pi)*radij,-0.2*radij-1:]
#    #pas = okence[radij:-radij,-0.2*radij-1]
#    pas = okence[1.5*radij:-0.5*radij,-0.2*radij-1]
#    #cv2.imshow("Prikaz", pas)
#    #cv2.waitKey(100)
#    pas = cv2.GaussianBlur(pas, (9,9), 0)
#    maxVrednost, void, lMaxOkence, void = cv2.minMaxLoc(pas)
#    print("maxVrednost: " + str(maxVrednost))
#    okence = cv2.circle(okence, 
#                        tuple([4*radij, int(1.5*radij+lMaxOkence[1])])
#                        , 0, 255, thickness=2)
#    #kot += 0.5*np.pi*(lMaxOkence[1]-radij)/(radij)
#    #kot = 0.5*np.pi*(lMaxOkence[1]-radij)/(radij)
#    kot = 0.5*np.pi*((np.float64(lMaxOkence[1])-0.5*radij)/radij)
#    print("kot: " + str(kot) + "\n")
#    lMaxKanal1 = [lMaxKanal1[0] + radij*np.cos(kot),
#                    lMaxKanal1[1] - radij*np.sin(kot)]
#    kanal1 = cv2.circle(kanal1, tuple([int(l) for l in lMaxKanal1]),
#                            2, 255, thickness=1, lineType=cv2.LINE_AA)
#    #cv2.imshow("Prikaz", okence)
#    #cv2.waitKey(100)
##kanal1 = cv2.circle(kanal1, tuple(lMaxKanal1), 0, 255, thickness=2)
#cv2.imshow("Prikaz", kanal1[:,:])
#print("shranjujem sliko v pikice.jpg")
#cv2.imwrite("pikice.jpg", kanal1)
#cv2.imshow("Prikaz", sl)
#cv2.waitKey(20000)


cv2.destroyAllWindows()




#void, slC = cv2.threshold(slC, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
