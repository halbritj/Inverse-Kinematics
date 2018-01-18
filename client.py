import socket


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 5050))

while True:
    data = input('\n>')
    sock.sendall(data.encode('utf-8') + b'\n')
    print(sock.recv(4096))
