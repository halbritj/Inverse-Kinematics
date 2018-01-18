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

goal = np.array([
    [  0.,   0.,   -1.,  -60.],
    [  0.,   1.,   0.,   0.],
    [  1.,   0.,   0.,  30.]], dtype=np.float32)

LMB = 1
MMB = 2
RMB = 3
SCROLL_DOWN = 4
SCROLL_UP = 5

keys = {LMB: False, MMB: False, RMB: False}



def map_sphere(p):
    r = 30
    L = np.linalg.norm( np.array(p) - 250)
    P = np.array([p[0] - 250, p[1] - 250, 0], dtype=np.float32)

    if L > 0.707106 * r: P[-1] = (r**2)/(2*L)
    else: P[-1] = np.sqrt(r**2 - L**2)

    return P

class view_port:
    def __init__(self):
        self.zoom = np.array([ 0.1,  0.1,  0.1], dtype=np.float32)
        self.trans = np.array([0, 8, -1.5], dtype=np.float32)
        self.rot = np.array([0, 0], dtype=np.float32)

        self.u = np.array([1,0,0], dtype=np.float32)
        self.theta = 0

    def event_handler(self, e):
        '''
        uses combination of shoemake and holroyds trackball methods
        http://www.diku.dk/~kash/papers/DSAGM2002_henriksen.pdf
        '''
        if e.type == QUIT:
            pygame.quit()
            exit()
        elif e.type == KEYDOWN:
            None
        elif e.type == MOUSEBUTTONDOWN:
            if e.button == SCROLL_UP: self.zoom += .1*self.zoom
            elif e.button == SCROLL_DOWN: self.zoom -= .1*self.zoom
            elif e.button == LMB:
                self.P_a = map_sphere(e.pos)
                print(self.P_a)
                keys[LMB] = True
        elif e.type == MOUSEBUTTONUP:
            if e.button == LMB:
                keys[LMB] = False
                #self.rot = np.copy(self.initial_rot)
        elif e.type == MOUSEMOTION:
            pos = np.array(e.pos, dtype=np.float32)
            if keys[LMB]:
                self.P_c = map_sphere(e.pos)

                self.u = np.cross(self.P_a, self.P_c)
                L = np.linalg.norm(self.u)
                self.theta = np.arctan2( np.linalg.norm(self.u), np.dot(self.P_a, self.P_c) )


                self.theta = np.rad2deg(self.theta)
                self.u /= L

                print(self.u, self.theta)
                
    def set_view(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 1, 0.1, 100)

        glRotatef(-90, 1, 0, 0)

        glRotatef(self.theta, *self.u)
        
        glTranslatef(*self.trans)
        glScalef(*self.zoom)

def loop(screen):
    clock = pygame.time.Clock()

    vao = grid()

    view = view_port()

    R = segment.robot(table, 'kuka_seg_%d')
    R.Shader = genShaders()
    R.shader()

    R.setGoal(goal)
 
    glUseProgram(R.Shader)
    while True:
        for e in pygame.event.get(): view.event_handler(e)
        view.set_view()
        
        clock.tick(60)
        #print(clock.get_fps())
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        glBindVertexArray(vao)
        glDrawArrays(GL_LINES, 0, 55)
        
        R.draw()

        pygame.display.flip()


if __name__ == "__main__":
    screen = init()
    r = loop(screen)
