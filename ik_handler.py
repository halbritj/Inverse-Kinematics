import random
import struct
import numpy as np
import time
import socket
from array_bytes import *

class IK:
    def __init__(self):
        self.min_theta = np.array(
            np.deg2rad([-185., -65., -220., -350., -130., -350.]), np.float32)

        self.max_theta = np.array(
            np.deg2rad([ 185., 125.,   64.,  350.,  130.,  350.]), np.float32)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(('localhost', 5050))


    def f(self): return self.communication('self.f(*self.theta)')

    def getArm(self): return self.communication('self.getArm()')

    def getTheta(self): return self.communication('self.theta')

    def communication(self, data):
        self.sock.sendall(b'0' + data.encode('utf-8') + b'\n')
        reply = self.sock.recv(4096)
        return bytes2array(reply)

    def setTheta(self, array):
        self.sock.sendall(b'1' + array2bytes(self.theta) + b'\n')
        self.sock.recv(4096)

    def loop(self, goal):
        self.goal = goal
        self.min_e = np.inf
        while self.min_e > 1:
            self.step()

    def step(self):
        self.theta = self.getTheta()

        T = self.getArm()

        e = self.goal - T[-1,:3]

        self.min_e = min(self.min_e, np.linalg.norm(e))

        start = time.time()
        #J = self.f(*self.theta) #get the jacobian for the end effector
        J = self.f()
        
        dur = time.time() - start
        #print(dur)

        m, n = J.shape #12 x n-joints

        Pn_0 = np.identity(n)
        free = np.ones(n, bool)
        clamping = True
        y = .1

        #print(free)

        while clamping:
            clamping = False

            
            Ji = np.dot( J.T , np.linalg.inv( np.dot(J, J.T) + y * np.identity(m) ))  #SDLS


            
            dTheta = np.dot(Ji, e.flatten())
        
            '''
            theta = self.theta[free]
            dTheta = dTheta[free]

            temp = theta + .05*dTheta

            ldx = temp < self.min_theta[free]
            temp[ldx] = self.min_theta[free][ldx]

            gdx = temp > self.max_theta[free]
            temp[gdx] = self.max_theta[free][gdx]

            clamping = (ldx|gdx).any()

            if clamping:
                
                cv = temp - theta
                J_ = J.T[ldx|gdx].reshape(3, 4)

                e -= np.sum(J_ * cv[:, None, None], axis=0)
                J.T[ldx|gdx] = 0

            self.theta[free] = temp
            free[ldx|gdx] = False

            '''
            for i in range(n):
                if not free[i]: continue
                
                temp = self.theta[i] + .01*dTheta[i]

                if temp < self.min_theta[i] or temp > self.max_theta[i]:
                    #print('clamping')
                    if temp > self.max_theta[i]: temp = self.max_theta[i]
                    elif temp < self.min_theta[i]: temp = self.min_theta[i]
                    
                    clamping = True

                    cv = temp - self.theta[i]

                    J_ = J.T[i]
                    e -= cv*J_.reshape(3,4)
                    J.T[i] = 0
                    Pn_0[i,i] = 0
                    
                    free[i] = False
                    
                self.theta[i] = temp

                self.setTheta(self.theta)




goal = np.array([
    [  0.,   0.,   -1.,  -60.],
    [  0.,   1.,   0.,   0.],
    [  1.,   0.,   0.,  30.]], dtype=np.float32)

ik = IK()
ik.loop(goal)

