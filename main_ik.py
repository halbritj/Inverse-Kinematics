import numpy as np
import socket

goal = np.array([
    [  0.,   0.,   -1.,  -60.],
    [  0.,   1.,   0.,   0.],
    [  1.,   0.,   0.,  30.]], dtype=np.float32)


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 5050))

while True:
    data = input('\n>')
    sock.sendall(data.encode('utf-8') + b'\n')
    print(sock.recv(4096))
 
