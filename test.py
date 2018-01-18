import server
import threading
import asyncore

a = 5

client = server.Server('localhost', 5050)

comm = threading.Thread(target=asyncore.loop)
comm.daemon = True
comm.start()
