Ta projekt zajema programsko opremo, napisano za moj magisterij. Gre se 
za nelinearno mehaniko kontituuma. Pomemben del je tudi strojni vid. 
Programski jezik je pretežno Python.

This project encapsulates the software written for my Master Thesis. 
It's mostly nonlinear continuum mechanics. Machine vision is an 
additional important part. The programming language used is Python.

Hyperelastic@LublanaJeBulana


POTEK:
1) s skripto kalibracija.py poiscemo parametre kamere.
    Ti se zapisejo v datoteki "paraParat/matrikaFotoaparata" in
    "paraParat/popacenjeFotoaparata"
2) izravnamo fotografije s "izravnavaLece.py"
3) poravnamo slike s "./poravnava.py"
4) poravnamo tezavne slike s "./poravnavaTezavnih.py"
5) poiscemo tocke s "./zajemTock.py" 
6) poiscemo aproksimacijo funkcije vzmeti s "./togost.py"
7) 
