import cv2
import numpy as np
from glob import glob
from operator import add

cv2.namedWindow("Prikaz", cv2.WINDOW_NORMAL)
imena = sorted(glob('./slikePoravnane/nelin/*.jpg'))

    

for ime in imena[28:38]:
    print(ime)
    sl = cv2.imread(ime)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(6,6))

    sl = cv2.cvtColor(sl, cv2.COLOR_BGR2HSV)
    sl = cv2.insertChannel(clahe.apply(cv2.extractChannel(sl, 2)), sl, 2)
    sl = cv2.cvtColor(sl, cv2.COLOR_HSV2BGR)
    sl = cv2.medianBlur(sl, 7)

    izKan1 = [800, 2470, 1490, 3660] #samo obarvanost, izrez
    slKan1 = cv2.extractChannel(sl, 1)[izKan1[0]:izKan1[1],izKan1[2]:izKan1[3]]

    izKan2 = [izKan1[1]-izKan1[0]-60, izKan1[1]-izKan1[0]-20, 20, 60] #za iskanje zacetka
    slKan2 = slKan1[izKan2[0]:izKan2[1], izKan2[2]:izKan2[3]]
    slKan2[slKan2<100] = 0; slKan2[slKan2>150] = 255

    krogi = cv2.HoughCircles(slKan2, cv2.HOUGH_GRADIENT, 2, 20, #krogi
                                param1=10, param2=10, minRadius=8, maxRadius=12)

    print(krogi[0,0])

    krogi = np.uint16(np.around(krogi))
    for k in krogi[0,:1]: #narise kroge
        cv2.circle(slKan1, (izKan2[2]+k[0], izKan2[0]+k[1]), k[2],
                                 155, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(slKan1, (k[0], izKan2[0]+k[1]), 1, 255, thickness=1)

    cv2.rectangle(slKan1, (izKan2[2], izKan2[0]), (izKan2[3], izKan2[1]), 255)

    cv2.imshow("Prikaz", slKan1[1300:,:-1900])
    cv2.waitKey(20000)


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
