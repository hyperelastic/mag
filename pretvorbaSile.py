#!/usr/bin/python2
import numpy as np
from glob import glob
from scipy import interpolate
import matplotlib.pyplot as plt
from string import split
np.set_printoptions(precision=1)
#np.set_printoptions(suppress=True)

meriloPovecave=1.5 # primerjaj paraParat/lokacijeKod in paraParat/lokacijeKodFizikalne
redAproksimacije = 6
imenaTabel = sorted(glob("tabele/[ln]*Korak*.txt"))
print(imenaTabel)

koef = np.loadtxt("./tabele/funkcijaTogosti.txt")
print(koef)

def sila(dolzina):
    dolzNicneSile = np.roots(koef)[1]
    sila = np.zeros_like(dolzina)
    sila[dolzina > dolzNicneSile] = \
         koef[0]*dolzina[dolzina>dolzNicneSile]**2 +\
         koef[1]*dolzina[dolzina>dolzNicneSile] + koef[2]
    return(sila)

for iT in imenaTabel[-1:]:
    print(iT)
    M = np.loadtxt(iT)
    razdX = M[1::2,0]; razdY = M[::2,0]
    razd = np.sqrt(razdX**2 + razdY**2)

    f = sila(razd)
    M[1::2,0] = f*razdX/razd
    M[::2,0] = -f*razdY/razd # minus zaradi obrata k.s.

    M[:,2:] /= meriloPovecave
    M[::2,2:] *= -1 # minus zaradi obrata k.s.
    for vrst in M[:,2:]: vrst -= vrst[0]  #nastavitev nicle
    M = np.insert(M, 2, 0, axis=1); M[2::2,3:] += 6 #zamik 6mm zaradi prve


    #popis s parametricnim polinomom 
    konstante = []
    t = np.cumsum(np.insert(
        np.sqrt(np.diff(M[::2,2:], axis=1)**2+
        np.diff(M[1::2,2:], axis=1)**2),
        0,0,axis=1), axis=1)
    for i in range(0,np.shape(t)[0], 1):
        print(t[i][-1])
        c = np.polyfit(t[i], M[2*i+1,2:], redAproksimacije)
        c[-1] = 0 #<- konstantni clen je 0
        d = np.polyfit(t[i], M[2*i,2:], redAproksimacije)
        d[-1] = 0 #<- konstantni clen je 0
        konstante.append(c); konstante.append(d)

        #prikaz v koordinatnem prostoru in shranjevanje slike
        x = np.polyval(c, t[i])
        y = np.polyval(d, t[i])
        graf = plt.plot(x, y, 'r-')
        plt.setp(graf, linewidth=0.2)
        plt.arrow(x[-1], y[-1], 100*M[2*i+1,0], 100*M[2*i,0],
            head_width=2, head_length=5, fc='g', ec='g',
            linewidth = 0.2)
    plt.title("MERITEV: polinomi-" + split(iT, "/")[-1])
    plt.xlabel("x [mm], Fx[100*N]"); plt.ylabel("y [mm], Fy[100*N]")
    plt.axis('equal')
    plt.xlim((0,1300)); plt.ylim((-200, 1000))
    plt.savefig(split(iT, "/")[0] + "/polinomiGrafi-" +
        split(iT, "/")[-1][:-4] + ".png", format = "png", dpi = 600/1.605)
    plt.show()
    plt.close()

    #shranjevanje tabele
    konstante = np.array(konstante)
    konstante = np.insert(konstante, [0,0], 0, axis=1) # vnese 2 nicli
    konstante[:,:2] = M[:,:2] # doda sile in case
    np.savetxt(split(iT, '/')[0]+"/polinomi-"+split(iT, '/')[-1],
        konstante[:,:-1],
        fmt = tuple(['%3.3f', '%i']+['%3.3e']*(redAproksimacije)),
        delimiter = " ",
        header = "\nMERITEV: polinomiNiz" + split(iT, "Niz")[-1] +
            "\n\nVsebuje sile v vzmeti za smeri x in y, case zajema slik " +
            "in konstante polinoma, popisujocega obliko nosilca.\n" +
            "Dolzina nosilca, torej najvecja vrednost parametra t je 1294mm\n" +
            "Koordinate y narascajo navzgor, " +
            "koordinate x pa desno.\n\nNosilec je popisan parametricno.\n" +
            "Dolocata ga dva polinoma (v x in y smeri):\n\tx[i](t) = " +
            "Kx[i](**N)*t**N + Kx[i](**N-1)*t**(N-1) + ... + Kx[i](2)*t**2 + "+
            "K[i](1)*t**1,\nkjer i oznacuje serijo zajema tock, N stopnjo " +
            "polinoma, Kx(**N) pa konstanto, ki pripada N-ti potenci.\n" +
            "Ustrezne oznake veljajo za Ky. Ker je konstanta " +
            "pri nicti potenci enaka nic, so shranjene le konstante potenc, " +
            "visjih od 1.\nKoordinate so metricne, v milimetrih. " +
            "Sile so v N. Casi so zapisani v obliki HHMMSS\n" +
            "\nFormat zapisa:\nSilaX[0],\tCasZajema[0]\tKx[0](**N),\t...,\t" +
            "Kx[0](**2),\tKx[0](**1)\nSilaY[0]\tCasZajema[0]\tKy[0](**N)," +
            "\t...,\tKy[0](**2),\tKy[0](**1)\nSilaX[1],\tCasZajema[1]\t"+
            "Kx[1](**N),\t...,\tKx[1](**2),\tKx[1](**1)\nSilaY[1]\t" +
            "CasZajema[1]\tKy[1](**N),\t...,\tKy[1](**2),\tKy[1](**1)" +
            "\n\t\t\t\t\t\t .\n\t\t\t\t\t\t .\n\t\t\t\t\t\t .\n")
                 

