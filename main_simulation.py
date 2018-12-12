from server import *
import threading
import asyncore

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders

import numpy as np

import segment

def init():
    pygame.init()
    display = (500, 500)
    screen = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    pygame.display.set_caption('KINEMATICS-OPENGL TEST')

    img = pygame.image.load('window_icon.png')
    pygame.display.set_icon(img)

    glClearColor(0.95, 1.0, 0.95, 0)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CCW)

    return screen

table = [[  0.        ,  -1.57079637,  10.23622036,  26.5748024 ],
         [ -1.57079637,   0.        ,  26.77165604,   0.        ],
         [  3.14159274,   1.57079637,   1.37795639,   0.        ],
         [  0.        ,  -1.57079637,   0.        ,  26.37795258],
         [  3.14159274,  -1.57079637,   0.        ,   0.        ],
         [  0.        ,   0.        ,   0.        ,   6.22047043]]

def grid():
    max_ = 15
    divs = 11
    A = np.linspace(-max_, max_, divs)

    UP = np.vstack((A, [max_]*divs, [0]*divs)).T
    DOWN = np.vstack((A, [-max_]*divs, [0]*divs)).T
    VERT = np.hstack((UP, DOWN)).astype(np.float32)

    LEFT = np.vstack(([-max_]*divs, A,  [0]*divs)).T
    RIGHT = np.vstack(([max_]*divs, A,  [0]*divs, )).T
    HORZ = np.hstack((LEFT, RIGHT)).astype(np.float32)

    LINES = np.vstack((HORZ, VERT))

    line_vao = glGenVertexArrays(1)
    glBindVertexArray(line_vao)

    line_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, line_vbo)
    glBufferData(GL_ARRAY_BUFFER, LINES.nbytes, LINES, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return line_vao

def genShaders():
    vertex = shaders.compileShader(*segment.vertex_shader)
    fragment = shaders.compileShader(*segment.fragment_shader)
    return shaders.compileProgram(vertex, fragment)

LMB = 1
MMB = 2
RMB = 3
SCROLL_DOWN = 4
SCROLL_UP = 5

keys = {LMB: False, MMB: False, RMB: False}

def map_sphere(p):
    w, h = pygame.display.get_surface().get_size()
    P = 2*(p/[w,h]) - 1

    r = 1/np.sqrt(2)

    R = np.array([-P[0], P[1], 0], np.float32)

    L = np.linalg.norm(P)

    if L <= r: R[2] = np.sqrt(r**2 - L**2)
    else: R[2] = r**2 / (2*L)

    return R
    

class view_port:
    def __init__(self):
        self.zoom = np.array([ 0.1,  0.1,  0.1], np.float32)
        self.trans = np.array([0,0,-10], np.float32)
        self.rot = np.array([0, 0], np.float32)

        self.u = np.array([1,0,0], np.float32)
        self.theta = 0

        self.start_rot = np.ndarray((4,4), np.float32)

    def event_handler(self, e):
        '''
        uses combination of shoemake and holroyds trackball methods
        http://www.diku.dk/~kash/papers/DSAGM2002_henriksen.pdf
        '''
        if e.type == QUIT:
            pygame.quit()
            exit()
        elif e.type == KEYDOWN:
            pass
        elif e.type == KEYUP:
            pass
        elif e.type == MOUSEBUTTONDOWN:
            if e.button == RMB: self.trans_start()
            elif e.button == MMB: self.rot_start()
            elif e.button == SCROLL_UP:  self.zoom += .1*self.zoom
            elif e.button == SCROLL_DOWN: self.zoom -= .1*self.zoom

        elif e.type == MOUSEBUTTONUP:
            if e.button == RMB: self.trans_set()
            elif e.button == MMB: self.rot_set()

        elif e.type == MOUSEMOTION:
            self.pos = np.array(e.pos, np.float32)
            if e.buttons == (0,0,1): self.trans_move()#RMB motion
            elif e.buttons == (0,1,0): self.rot_move() #MMB motion

    def trans_start(self):
        self.start_pos = np.copy(self.pos)
        self.start_trans = np.copy(self.trans)
        
    def trans_move(self):
        dx, dy = self.pos - self.start_pos
        self.trans = self.start_trans + np.array([dx,-dy,0], np.float32)/40
        
    
    def trans_set(self): pass

    def rot_start(self):
        self.P_a = map_sphere(self.pos)
        glGetFloatv(GL_MODELVIEW_MATRIX, self.start_rot)
        
    def rot_move(self):
        self.P_c = map_sphere(self.pos)
        
        self.u = np.cross(self.P_a, self.P_c)
        self.theta = np.arctan2( np.linalg.norm(self.u), np.dot(self.P_a, self.P_c) )

        self.theta = np.rad2deg(self.theta)
        self.u /= -np.linalg.norm(self.u)

        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(self.start_rot)
        glRotatef(self.theta, *self.u)
        
        
    def rot_set(self):
        #glMatrixMode(GL_MODELVIEW)
        #glRotatef(self.theta, *self.u)
        pass

    
    def set_view(self):

        #glMatrixMode(GL_MODELVIEW)
        #glLoadIdentity()

        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 1, 0.1, 100)
        
        glTranslatef(*self.trans)
        glScalef(*self.zoom)
        

def loop(screen):
    clock = pygame.time.Clock()

    vao = grid()

    view = view_port()

    R = segment.robot(table, 'kuka_seg_%d')
    R.Shader = genShaders()
    R.shader()

    S = Server('localhost', 5050, R)

    t = segment.Text('')

    glUseProgram(R.Shader)
    while True:
        
        for e in pygame.event.get(): view.event_handler(e)
        view.set_view()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        clock.tick(300)

        R.draw()
        
        glBindVertexArray(vao)
        glDrawArrays(GL_LINES, 0, 55)

        t.genText('FPS: %d' %(clock.get_fps()))
        t.draw()
        
        pygame.display.flip()


if __name__ == "__main__":
    screen = init()    
    r = loop(screen)
