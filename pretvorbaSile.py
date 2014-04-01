import numpy as np
from glob import glob
from scipy import interpolate
import matplotlib.pyplot as plt
np.set_printoptions(precision=1)
np.set_printoptions(suppress=True)

meriloPovecave=1.5 # primerjaj paraParat/lokacijeKod in paraParat/lokacijeKodFizikalne
imenaTabel = sorted(glob("tabele/*Korak*.txt"))
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

for iT in imenaTabel[-2:-1]:
    M = np.loadtxt(iT)
    razdX = M[1::2,0]; razdY = M[::2,0]
    razd = np.sqrt(razdX**2 + razdY**2)

    f = sila(razd)
    M[1::2,0] = f*razdX/razd
    M[::2,0] = -f*razdY/razd # minus zaradi obrata k.s.

    M[:,2:] /= meriloPovecave
    M[::2,2:] *= -1 # minus zaradi obrata k.s.
    print(M[:10,:7])
    for vrst in M[::2,2:]: vrst -= vrst[0]
    for vrst in M[1::2,2:]: vrst -= vrst[0]
    print(M[:10,:7])

    
    #test dolzine nosilca
    #for i in range(10):
    #    print(np.sum(np.sqrt(np.diff(M[i:i+2:2,2:])**2+np.diff(M[i+1:i+2:2,2:])**2)))

    dtPovp = np.average(np.sqrt(np.diff(M[::2,2:])**2+np.diff(M[1::2,2:])**2))
    print(dtPovp)
    #t = np.linspace(0, dtPovp*(np.shape(M[0,2:])[0]),
    #                num=np.shape(M[0,2:])[0])
    #print(t)
    tck = interpolate.splrep(M[1:2:2,2:][0], M[:2:2,2:][0], s=7)
    x = np.linspace(0, M[1,-1], np.shape(M[1,2:])[0])
    print(x)
    y = interpolate.splev(x, tck)
    print(y)
    #graf s tockami #    tocke = np.array(tocke)
    #plt.plot(M[1:2:2,2:], M[:2:2,2:], 'r.', x[2:], np.diff(np.diff(y)), 'b-')
    plt.plot(x[2:], np.diff(np.diff(y)), 'b-')
    plt.show()

    

##shranjevanje tabele
#np.savetxt(imeTabele[:-4] + "Niz" + str(stNiza) +
#    "Korak" + str(korak) + imeTabele[-4:], vsiPodatki,
#    fmt=tuple(["%.2f", "%i"] + ["%.2f" for i in\
#        range(np.shape(vsiPodatki)[1]-2)]),
#    delimiter=" ",
#    header= "\nDokument z dolzinami vzmeti, casi zajemov in tockami.\n\n" +
#            "Obdelane so slike v obsegu od '" + imena[0] +
#            "' do '" + imena[-1] + "'.\n\n" +
#            "Podatki so shranjeni v sledeci obliki:\n" +
#            "DolzinaVzmetiY[0], Cas[0], TockeY[0]\n" +
#            "DolzinaVzmetiX[0], Cas[0], TockeX[0]\n" +
#            "DolzinaVzmetiY[1], Cas[1], TockeY[1]\n" +
#            "DolzinaVzmetiX[1], Cas[1], TockeX[1]\n" +
#            "\t...\n"+
#            "DolzinaVzmetiY[:], Cas[:], TockeY[:]\n" +
#            "DolzinaVzmetiX[:], Cas[:], TockeX[:]\n\n" + 
#            "Koordinate x narascajo desno, y pa navzdol.\n" +
#            "Ce je 'DolzinaVzmeti' 10000.0, je bil nosilec povesen " +
#            "preko meja slike in ni bilo mogoce zajeti vseh tock.\n" +
#            "Nezajete tocke imajo koordinate 10000.0.\n\n"
#            )

