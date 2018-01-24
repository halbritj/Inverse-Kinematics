import random
import struct
import numpy as np
import time
import socket
from array_bytes import *
import matplotlib.pyplot as plt

class IK:
    def __init__(self):
        self.MIN_THETA = np.array(
            np.deg2rad([-185., -65., -220., -350., -130., -350.]), np.float32)

        self.MAX_THETA = np.array(
            np.deg2rad([ 185., 125.,   64.,  350.,  130.,  350.]), np.float32)

        self.min_theta = np.empty_like(self.MIN_THETA)
        self.max_theta = np.empty_like(self.MAX_THETA)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(('localhost', 5050))

        self.terminator = b'\r\n\r\n'


    def f(self): return self.communication('self.f(*self.theta)')

    def getArm(self): return self.communication('self.getArm()')

    def getTheta(self): return self.communication('self.theta')

    def communication(self, data):
        self.sock.sendall(b'0' + data.encode('utf-8') + self.terminator)
        reply = self.sock.recv(4096)
        return bytes2array(reply)

    def setTheta(self):
        self.sock.sendall(b'1' + array2bytes(self.theta) + self.terminator)
        self.sock.recv(4096)

    def setTurn(self, turn):
        self.min_theta[:] = self.MIN_THETA
        self.max_theta[:] = self.MAX_THETA
        
        for i in range(6):
            if (turn & 1<<i): #get ith bit
                self.max_theta[i] = 0
            else:
                self.min_theta[i] = 0
            

    def loop(self, goal, max_error=10**-1):
        
        '''
        https://link.springer.com/content/pdf/10.1007/s00371-004-0244-4.pdf
        http://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/SdlsPaper.pdf (selectively damped least squares)
        http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4309644

        consider STATUS and TURN sequences
        http://www.hgpauction.com/wp-content/uploads/2017/02/Expert-Programming-5.2.pdf

        Value   Bit 5   Bit 4   Bit 3   Bit 2   Bit 1   Bit 0
        0       A6≥0    A5≥0    A4≥0    A3≥0    A2≥0    A1≥0
        1       A6<0    A5<0    A4<0    A3<0    A2<0    A1<0
        
        '''

        self.setTurn(0b010001)
        
        error = []
        data = np.ndarray((6,0), np.float32)
        
        self.theta = np.zeros([6], np.float32)
        self.setTheta()
        self.goal = goal
        self.min_e = np.inf
        while self.min_e > max_error: #convergence loop
            self.theta = self.getTheta()

            T = self.getArm()

            e = self.goal - T[-1,:3]

            self.min_e = min(self.min_e, np.linalg.norm(e))


            J = self.f() #get the jacobian for the end effector
            

            m, n = J.shape #12 x n-joints

            Pn_0 = np.identity(n)
            free = np.ones(n, bool)
            clamping = True
            y = .1

            S = np.zeros((J.shape))

            a = (self.min_theta + self.max_theta) / 2

            #print('clamping')
            while clamping: #clamping loop
                clamping = False

                u,s,v = np.linalg.svd(J)
                
                S[:s.size,:s.size] = np.diag(1/s) #svd
                J_SVD = np.dot(u, np.dot(S, v)).T

                S[:s.size,:s.size] = np.diag(s/(s**2 + y)) #sdls
                J_SDLS = np.dot(u, np.dot(S, v)).T

                Pn = Pn_0 - np.dot(J_SVD, J)

                h = (1/6)*np.sum(((self.theta - a)/(a - self.max_theta))**2)

                #print(h)
                
                dTheta = np.dot(J_SDLS, e.flatten()) - (h*.001)*Pn_0.diagonal()
            
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

                    self.setTheta()
                    data = np.hstack((data, self.theta[:, None]))
                error.append(self.min_e)


        plt.figure()
        
        sub = plt.subplot(711)
        sub.plot(np.array(error))
        
        for i, theta in enumerate(data):
            sub = plt.subplot('71%d' %(i+2))
            sub.plot(theta)
            
                    
        print('goal met with error %.3f' %(self.min_e))
        plt.show()





goal = np.array([
    [  0.,   0.,   -1.,  -60.],
    [  0.,   1.,   0.,   0.],
    [  1.,   0.,   0.,  30.]], dtype=np.float32)





ik = IK()
'''
for i in np.linspace(0, 2*np.pi, 30):
    dy, dz = 30*np.array([np.cos(i), np.sin(i)])
    goal = np.array([
        [  0.,   0.,   1.,  30.],
        [  1.,   0.,   0.,   0. + dy],
        [  0.,   1.,   0.,  40. + dz]], dtype=np.float32)
'''
ik.loop(goal)

