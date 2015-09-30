

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
        duxdx = self.compute_gradient(
            u[:,:,0], self.h, self.h,
            #edge_order=1
        )[:,:,0]
        duydx = self.compute_gradient(
            u[:,:,1], self.h, self.h,
            #edge_order=1
        )[:,:,0]
        duxdy = self.compute_gradient(
            u[:,:,0], self.h, self.h,
            #edge_order=1
        )[:,:,1]
        duydy = self.compute_gradient(
            u[:,:,1], self.h, self.h,
            #edge_order=1
        )[:,:,1]
        
        s = u.copy()
        s[:,:,0] = u_x * duxdx + u_y * duxdy
        s[:,:,1] = u_x * duydx + u_y * duydy
        
        s[[0, -1], :, :] = 0
        s[:, [0, -1], :] = 0
        return s
    
    def perform_advection(self, w1, dt):
        w2 = w1.copy()
        w2 = w1 - dt * self.advection_primitive(w1)
        return w2

    def perform_diffusion(self, w2, dt):
        w3 = w2 + dt * self.viscosity * self.compute_laplacian(w2, self.h, self.h)
        return w3
