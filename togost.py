
import numpy as np
import matplotlib.pyplot as plt


g = 0.00981 #[0.001*m/s^2]
x0 = np.loadtxt('./tabele/vzmet2.txt')[:,2]
y0 = np.loadtxt('./tabele/vzmet2.txt')[:,1]*g

koef = np.polyfit(x0, y0, 2)
print("Polinom aproksimacije je podan z enacbo:\nsila[N] = " +
        str(koef[0]) + "*dolzina[px]^2 + " + str(koef[1]) +
        "*dolzina[px] + " + str(koef[2]) + ".")

x1 = np.linspace(np.roots(koef)[1], np.max(x0), num=100)
y1 = koef[0]*x1**2 + koef[1]*x1**1 + koef[2]


np.savetxt("./tabele/funkcijaTogosti.txt", koef,
    fmt=("%e"), delimiter=" ",
    header= "\nDokument s priblizno funkcijo togosti vzmeti " +
            "za podrocje sil, vecjih od 0N. Funkcija je oblike:\n" +
            "sila[N] = koef[0]*dolzina[px]^2 + " +
            "koef[1]*dolzina[px] + koef[2].\n"
            "Dolzino pri sila=0 lahko dobis s np.roots(koef)[1]\n")

plt.plot(x0, y0, 'ro')
plt.plot(x1, y1, 'g-')
plt.xlabel('Dolzina vzmeti, slikovne tocke')
plt.ylabel('Sila v vzmeti, N')
plt.show()








