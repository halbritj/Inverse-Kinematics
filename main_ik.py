import numpy as np
import socket
from array_bytes import *
import time
goal = np.array([
    [  0.,   0.,   -1.,  -60.],
    [  0.,   1.,   0.,   0.],
    [  1.,   0.,   0.,  30.]], dtype=np.float32)


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 5050))

a = np.array([1,2,3,4,5,6], np.float32)

print(array2bytes(a))

#while True:
#data = input('\n>')
#sock.sendall(data.encode('utf-8') + b'\n')
sock.sendall(b'1' + array2bytes(a) + b'\n')

#sock.sendall('0self.theta'.encode('utf-8') + b'\n')

reply = sock.recv(4096)
print(bytes2array(reply))
