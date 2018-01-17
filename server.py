import asyncore
import asynchat
import socket
import threading

class ChatHandler(asynchat.async_chat):
    def __init__(self, sock):
        asynchat.async_chat.__init__(self, sock=sock)
 
        self.set_terminator(b'\n')
        self.buffer = []
 
    def collect_incoming_data(self, data):
        self.buffer.append(data)
 
    def found_terminator(self):
        print('Received:', self.buffer)
        self.buffer = []

class Server(asyncore.dispatcher):
 
    def __init__(self, host, port):
        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.bind((host, port))
        self.listen(5)
        
    def handle_accept(self):
        pair = self.accept()
        if pair is not None:
            sock, addr = pair
            print('Incoming connection from %s' % repr(addr))
            handler = ChatHandler(sock)


client = Server('localhost', 5050)

comm = threading.Thread(target=asyncore.loop)
comm.daemon = True
comm.start()
