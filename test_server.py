from server import *
import threading
import asyncore

class cat:
    def __init__(self):
        pass

    def communication(self, data):
        return data + b'meow'



S = Server('localhost', 5050)

S.client = cat()

comm = threading.Thread(target=asyncore.loop)
comm.daemon = True
comm.start()
