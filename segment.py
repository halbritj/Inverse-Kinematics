import cv2
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders

import DH_METHODS as DHM
from read_obj import OBJ

import time
from array_bytes import *

dtypes = {
    np.dtype('float32'):    GL_FLOAT,
    np.dtype('uint8'):      GL_UNSIGNED_BYTE,
    np.dtype('uint16'):     GL_UNSIGNED_SHORT
    }

VECTOR_ATTRIB = 0
TEXTURE_ATTRIB = 1
COLOR_ATTRIB = 1
NORMAL_ATTRIB = 2



class frame:
    def __init__(self, T):

        T[:3, :3] *= 10
        
        u,v,w = (T[:3, :3] + T[:3, -1:]).T
        p = T[:3, -1]
        
        lines = np.vstack((p,u,p,v,p,w)).astype(np.float32)

        colors = np.array([
            [255,0,0],
            [255,0,0],
            [0,255,0],
            [0,255,0],
            [0,0,255],
            [0,0,255]], dtype=np.uint8)
        
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        bindBuffer(GL_ARRAY_BUFFER, lines, GL_STATIC_DRAW, VECTOR_ATTRIB)
        bindBuffer(GL_ARRAY_BUFFER, colors, GL_STATIC_DRAW, COLOR_ATTRIB)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        
    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_LINES, 0, 6)


def bindTexture(img):
    h, w, _ = img.shape
    
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(GL_TEXTURE_2D,
                    0,
                    GL_RGB,
                    w, h,
                    0,
                    GL_BGR,
                    dtypes[img.dtype],
                    img)

    glBindTexture(GL_TEXTURE_2D, 0)
    
    return texture

def bindBuffer(target, array, usage, index, normalized=GL_FALSE, stride=0):
    buffer = glGenBuffers(1)
    glBindBuffer(target, buffer)
    glBufferData(target, array.nbytes, array, usage)
    glVertexAttribPointer(index, array.shape[-1], dtypes[array.dtype], normalized, stride, None)
    glEnableVertexAttribArray(index)

class segment:
    def __init__(self, name):
        self.img = cv2.imread('data\%s.png' %(name))

        self.v, self.t, self.n = OBJ('data\%s.obj' %(name))

        self.texture = bindTexture(self.img)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        bindBuffer(GL_ARRAY_BUFFER, self.v, GL_STATIC_DRAW, VECTOR_ATTRIB)
        bindBuffer(GL_ARRAY_BUFFER, self.t, GL_STATIC_DRAW, TEXTURE_ATTRIB)
        bindBuffer(GL_ARRAY_BUFFER, self.n, GL_STATIC_DRAW, NORMAL_ATTRIB)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def draw(self):
        #T represents the transformation from the base frame
        glBindTexture(GL_TEXTURE_2D, self.texture)
        #glUniform1i(loc, 0)
        
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.v.size//3)


class robot:
    def __init__(self, table, name):
        self.size = len(table)
        self.table = table
        
        self.baseFrame = np.identity(4)

        self.theta = np.zeros(self.size, dtype=np.float32)

        self.DH, self.f = DHM.createParams(table)
        #self.DH = [self.baseFrame] + self.DH
        self.segments = []

        for i in range(self.size + 1):
            print(name %(i))
            seg = segment(name %(i))
            self.segments.append(seg)

        self.frame = frame(np.identity(4))

    def getArm(self):
        T = np.zeros((self.size+1, 4, 4), dtype=np.float32)
        T[0] = self.baseFrame
        for i in range(self.size):
            T[i+1] = np.dot( T[i], self.DH[i](self.theta[i]) )
            #T[i+1] = self.DH[i]( self.theta[i] )
        return T

    def shader(self):
        glBindAttribLocation(self.Shader, VECTOR_ATTRIB, 'position')
        glBindAttribLocation(self.Shader, TEXTURE_ATTRIB, 'texture')
        glBindAttribLocation(self.Shader, NORMAL_ATTRIB, 'normal')
        #loc = glGetUniformLocation(self.Shader, 'myTextureSampler')

    '''
    def move(self, dTheta):
        self.theta += dTheta

        gdx = self.theta > self.max_theta
        self.theta[gdx] = self.max_theta[gdx]

        ldx = self.theta < self.min_theta
        self.theta[ldx] = self.min_theta[ldx]

        return ldx, gdx
    '''
    
    def draw(self):

        T = self.getArm()

        glMatrixMode(GL_MODELVIEW)
        modelview = glGetFloatv(GL_MODELVIEW_MATRIX)

        for i, seg in enumerate(self.segments):
            glLoadMatrixf(T[i].T)
            seg.draw()

        glLoadMatrixf(T[-1].T)
        self.frame.draw()

        glLoadMatrixf(modelview)

        #self.step(T[-1]) #move the robot to its next position

    def communication(self, data):
        print(data)
        if data[0] == 48: #get variable
            command = data[1:].decode('utf-8')
            array = eval(command)
            return array2bytes(array)
        elif data[0] == 49: #set variable
            print('here')
            self.theta = bytes2array(data[1:])
            print(self.theta)
            return b'\n'
            


vertex_shader = ("""
    #version 140
    uniform mat4 gl_ProjectionMatrix;
    uniform mat4 gl_ModelViewMatrix;
    in vec3 position;
    in vec2 texture;

    out vec2 UV;

    void main(void)
    {
        gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vec4(position, 1.0);
        UV = texture;
    }
    """, GL_VERTEX_SHADER)

fragment_shader = ("""
    #version 140
    in vec2 UV;
    out vec3 Color;
    uniform sampler2D myTextureSampler;
    void main(void)
    {
        //float i = dot( fNormal.xyz, vec3(1.0,1.0,0) );
        //Color = vec3(.1, 0, .5);
        //Color = normalize( vec4(i,i,i,1.0) );
        //Color = vec4(1.0, .3, .1, 1.0);
        Color = texture( myTextureSampler, UV ).rgb;
    }
    """, GL_FRAGMENT_SHADER)


