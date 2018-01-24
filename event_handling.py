
class event_tree:
    def __init__(self):
        self.active = []

        

class event_pair(tuple):
    def __new__(self, event, callback=None):
        return tuple.__new__(self, (event, callback))
        





def method(event):
    print(event)

a = [event_pair('a'), event_pair('b', method), event_pair('c', method)]

for key, callback in a:
    if callback: callback(key)
