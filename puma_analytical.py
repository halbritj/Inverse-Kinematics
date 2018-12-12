import numpy as np
from DH_METHODS import allDH


table = np.array([
    [   1.57079633,   -1.57079633,    0.        ,    0.        ],
    [   0.        ,    0.        ,  431.8       ,  149.09      ],
    [   1.57079633,    1.57079633,  -20.32      ,    0.        ],
    [   0.        ,   -1.57079633,    0.        ,  433.07      ],
    [   0.        ,    1.57079633,    0.        ,    0.        ],
    [   0.        ,    0.        ,    0.        ,   56.25      ]])

min_theta = np.array([
    -160,
    -225,
    -45,
    -110,
    -100,
    -266])

max_theta = np.array([
    160,
    45,
    225,
    170,
    100,
    266])

def getArm(DH, theta):
    T = np.zeros((6, 4, 4), float)
    t = np.identity(4)
    for i in range(6):
        T[i] = t.dot( DH[i](theta[i]) )
        t = T[i]
    return T



theta = np.zeros(6, float)
A = allDH(table)

T = getArm(A, theta)


ARM = -1
ELBOW = -1
WRIST = +1
FLIP = -1

#px, py, pz = T[-1, :3, 3] - table[-1, -1]*T[-1, :3, 2]
px, py, pz = T[3, :3, 3]

a2 = table[1, 2]
d2 = table[1, 3]
d4 = table[3, 3]
a3 = table[2, 2]



R = np.sqrt(px**2 + py**2)
r = np.sqrt(px**2 + py**2 - d2**2)
dy = -ARM * py * r - px * d2
dx = -ARM * px * r + py * d2
theta_1 = np.arctan2(dy, dx)



R = np.sqrt(px**2 + py**2 + pz**2 - d2**2)
r = np.sqrt(px**2 + py**2 - d2**2)

sin_a = -pz/R
cos_a = -(ARM*r)/R
cos_b = (a2**2 + R**2 - d4**2 - a3**2)/(2*a2*R)
sin_b = np.sqrt(1 - cos_b**2)

dy = sin_a * cos_b + (ARM*ELBOW) * cos_a * sin_b
dx = cos_a * cos_b - (ARM*ELBOW) * sin_a * sin_b

theta_2 = np.arctan2(dy, dx)



