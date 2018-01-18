import asyncore
import asynchat
import socket
import threading

'''http://www.grantjenks.com/wiki/random/python_asynchat_chat_example'''

class ChatHandler(asynchat.async_chat):
    def __init__(self, sock):
        asynchat.async_chat.__init__(self, sock=sock)

        self.sock = sock        
        self.set_terminator(b'\n')
        self.buffer = []
 
    def collect_incoming_data(self, data):
        self.buffer.append(data)
 
    def found_terminator(self):
        reply = self.client.communication(self.buffer.pop())
        #print('Received:', self.buffer)
        self.sock.send(reply)
        #self.buffer = []

class Server(asyncore.dispatcher):
 
    def __init__(self, host, port, client):
        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.bind((host, port))
        self.listen(5)

        self.client = client
        comm = threading.Thread(target=asyncore.loop)
        comm.daemon = True
        comm.start()
        
    def handle_accept(self):
        pair = self.accept()
        if pair is not None:
            sock, addr = pair
            print('Incoming connection from %s' % repr(addr))
            handler = ChatHandler(sock)
            handler.client = self.client

