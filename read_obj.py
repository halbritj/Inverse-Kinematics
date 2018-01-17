import numpy as np
import re


def add(array, line):
    data = re.findall('(?<= )[-.\d]*(?=[ \n])', line)
    array.extend(data)



def OBJ(filename):
    file = open(filename, 'r')
    v = []
    t = []
    n = []
    f = []
    for line in file:
        if 'v ' in line: add(v, line)
        elif 'vt ' in line: add(t, line)
        elif 'vn ' in line: add(n, line)
        elif 'f ' in line:
            verts = re.findall('(?<= )[/\d]*(?=[ \n])', line)
            for vert in verts:
                f.extend(vert.split('/'))

    v = np.array(v, dtype=np.float32).reshape(-1, 3)
    t = np.array(t, dtype=np.float32).reshape(-1, 2)
    n = np.array(n, dtype=np.float32).reshape(-1, 3)

    f = np.array(f, dtype=np.uint16).reshape(-1, 3, 3) - 1

    t[:, 1] = 1 - t[:, 1]
    
    return v[f[:,:,0]], t[f[:,:,1]], n[f[:,:,2]]


