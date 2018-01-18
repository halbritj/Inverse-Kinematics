import random
import struct
import numpy as np
import time
from array_bytes import *

class IK:
    def __init__(self, goal):

        self.reset(goal)

    def reset(self, goal):
        self.min_e = np.inf
        self.goal = goal
        

    def f(self, theta):
        pass

    def DH(self, theta):
        pass




    def setGoal(self, goal):
        self.min_e = np.inf
        self.goal = goal
            
    def step(self, T):
        
        e = self.goal - T[:3]

        self.min_e = min(self.min_e, np.linalg.norm(e))

        start = time.time()
        J = self.f(*self.theta) #get the jacobian for the end effector
        dur = time.time() - start
        #print(dur)


        print(J.dtype)

        m, n = J.shape #12 x n-joints

        Pn_0 = np.identity(n)
        free = np.ones(n, dtype=bool)
        clamping = True
        y = .1


        #print(free)

        #asdfasdf
        
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
