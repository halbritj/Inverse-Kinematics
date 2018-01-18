import struct
import numpy as np

    
def array2bytes(array):
    dtype = np.sctype2char(array.dtype).encode('utf-8')
    return dtype + struct.pack('%dB' %(array.ndim+1), array.ndim, *array.shape) + array.tobytes()

def bytes2array(data):
    ndim = data[1]
    shape = struct.unpack('%dB' %(ndim), data[2:2+ndim])
    return np.fromstring(data[2+ndim:], chr(data[0])).reshape(shape)




if __name__ == '__main__':
    a = np.ndarray((5,2), np.uint64)
    b = array2bytes(a)
    c = bytes2array(b)
    print(np.all(a==c))

