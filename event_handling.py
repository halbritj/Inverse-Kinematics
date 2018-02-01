import pygame
from pygame.locals import *
from collections import deque
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders

LMB = 1
MMB = 2
RMB = 3
SCROLL_DOWN = 4
SCROLL_UP = 5

def init():
    pygame.init()
    display = (500, 500)
    screen = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    pygame.display.set_caption('KINEMATICS-OPENGL TEST')

    img = pygame.image.load('window_icon.png')
    pygame.display.set_icon(img)
    
'''
class tree(dict):
    def __init__(self, data=[]):
        dict.__init__(self)

        self.order = []
        self.active = {}
        self.motion = None
        
        for chain in data:
            self.add(chain)
        
    def add(self, chain, parent = None):
        self.parent = parent
        if chain:
            event = chain[0]
            if not event in self: self[event] = tree()
            self[event].add(chain[1:], event)

    def handler(self, event):
        if event.type == KEYDOWN:
            self.active[event.key] = event
            print(event)
        elif event.type == KEYUP:
            self.active.pop(event.key)
        elif event.type == MOUSEBUTTONDOWN:
            self.active[event.button] = event
        elif event.type == MOUSEBUTTONUP:
            self.active.pop(event.button)
        elif event.type in {MOUSEMOTION}:
            self.motion = event
'''

class event_handler(dict):
    def __init__(self):
        super().__init__()
        self.order = []

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.order.append(key)

    def pop(self, k, *args):
        super().pop(k, *args)
        self.order.remove(k)

    def handler(self, event):
        if event.type == KEYDOWN:
            self[event.key] = event   
        elif event.type == KEYUP:
            self.pop(event.key)
        elif event.type == MOUSEBUTTONDOWN:
            self[event.button] = event
            if event.button == RMB:
                self.start_zoom()
        elif event.type == MOUSEBUTTONUP:
            self.pop(event.button)
            if event.button == RMB:
                self.set_zoom()
        elif event.type in {MOUSEMOTION}:
            self.motion = event
            if event.buttons == (0,0,1):
                self.move_zoom()

    def start_zoom(self):
        self.start_pos = np.array(self.motion.pos)
        print(self.start_pos)

    def move_zoom(self):
        pass

    def set_zoom(self):
        final_pos = np.array(self.motion.pos)
        print(self.start_pos, final_pos, final_pos - self.start_pos)

            

class event_pair:
    def __init__(self, event, callback=None):
        self.event = event
        self.callback = callback

    def __repr__(self):
        return self.event

    def __iter__(self):
        return iter((self.event, self.callback))

    def __eq__(self, value):
        return self.event == value

    def __hash__(self):
        return hash(self.event)

def method(event):
    print(event)

def dummy(event):
    print('i overwrote your method')


hotkeys = [
    [event_pair('ctrl', dummy), event_pair('shift'), event_pair('s', method)], #save
    [event_pair('ctrl'), event_pair('c', method)], #copy
    [event_pair('shift'), event_pair('tab', method)] #tab
    ]

#t = tree(hotkeys)
t = event_handler()

queue = deque(['ctrl', 'c'])

chain = t
'''
while queue:
    event = queue.popleft()
    subTree = chain[event]
    if not subTree:
        subTree.parent.callback(event)
        break
    chain = subTree
'''    

screen = init()
font = pygame.font.SysFont('Arial', 15)
textSurface = font.render('asdf', True, (255,255,255,255), (0,0,0,255))
textData = pygame.image.tostring(textSurface, "RGBA", True)

glRasterPos2d( 0, 0)
glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

glRasterPos2d( .9, .9)
glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

pygame.display.flip()

while True:
    for e in pygame.event.get():
        if e.type == QUIT: pygame.quit()
        t.handler(e)



