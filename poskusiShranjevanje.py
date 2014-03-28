
from io import BytesIO
import numpy as np
np.set_printoptions(precision=2)




##########NACRT STRUKTURE DATOTEKE########## 
# RAZDALJEx[0]  CASI[0]  TOCKEx[0,:]
# RAZDALJEy[0]  CASI[0]  TOCKEy[0,:]
# RAZDALJEx[1]  CASI[1]  TOCKEx[1,:]
# RAZDALJEy[1]  CASI[1]  TOCKEy[1,:]
#            ... 
# RAZDALJEx[:]  CASI[:]  TOCKE[:,:]
# RAZDALJEy[:]  CASI[:]  TOCKE[:,:]




casi = np.array(range(5))
casi = np.repeat(casi, 2)
tocke = np.random.rand(10, 10)

print(casi); print(tocke) 
v = np.hstack((casi[:,None],tocke))

t = BytesIO()

#print(["%d"] + ["%.3f" for i in range(np.shape(tocke)[1])])
np.savetxt(t, v,
    fmt=tuple(["%d"] + ["%.2f" for i in range(np.shape(tocke)[1])]),
    delimiter=" ",
    header="Tu so shranjeni podatki v sledeci obliki:\n" +
            "A[0], B[0, 0], B[0, 1], ..... , B[0, M-1]\n"+
            "A[1], B[1, 0], B[1, 1], ..... , B[1, M-1]\n"+
            "\t...\n"+
            "A[:], B[:, 0], B[:, 1], ..... , B[:, M-1]"
            )
print t.getvalue()





