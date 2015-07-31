

import numpy as np
import numpy.linalg 

from .CFDSimulator import *

import scipy.interpolate as interpolate
import scipy.sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt 

import os.path

class CFDExplicitSimulator(CFDSimulator):
    
    def advection_primitive(self, u):
        u_x = u[:,:,0]
        u_y = u[:,:,1]
        duxdx = self.compute_gradient(u[:,:,0], self.h, self.h)[:,:,0]
        duydx = self.compute_gradient(u[:,:,1], self.h, self.h)[:,:,0]
        duxdy = self.compute_gradient(u[:,:,0], self.h, self.h)[:,:,1]
        duydy = self.compute_gradient(u[:,:,1], self.h, self.h)[:,:,1]
        
        dudx = u.copy()
        dudy = u.copy()
        dudx[:,:,0] = duxdx
        dudx[:,:,1] = duydx
        dudy[:,:,0] = duxdy
        dudy[:,:,1] = duydy
        dudy = np.array([duxdy, duydy])
        s = u.copy()
        s[:,:,0] = u_x * duxdx + u_y * duxdy
        s[:,:,1] = u_x * duydx + u_y * duydy
        return s
    
    def perform_advection(self, w1, dt):
        w2 = w1.copy()
        w2 = w1 - dt * self.advection_primitive(w1)
        return w2

    def perform_diffusion(self, w2, dt):
        w3 = self.diffusion(w2, dt)
        return w3
