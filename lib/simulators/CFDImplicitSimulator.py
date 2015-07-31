

import numpy as np
import numpy.linalg 

from .CFDSimulator import *

import scipy.interpolate as interpolate
import scipy.sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt 

import os.path

class CFDImplicitSimulator(CFDSimulator):
    

    def advection(self, w1, dt, path):
        w2 = w1.copy()
        psi = np.clip(path[:,:,1], 0, self.n).astype(int)
        psj = np.clip(path[:,:,0], 0, self.m).astype(int)
        w2[psi,psj,0] = w1[:, :, 0]
        w2[psi,psj,1] = w1[:, :, 1]
        return w2
        
    def update_path(self, path, w1, dt):
        # set new paths 
        newpath = path.copy()
        for i in range(int(self.n)):
            for j in range(int(self.m)):
                if not (path[i,j,0] < 0 or path[i,j,0] >= self.m or path[i,j,1] < 0 or path[i,j,1] >= self.n):
                    newpath[i,j,0] += dt*w1[path[i,j,1], path[i,j,0], 0]
                    newpath[i,j,1] += dt*w1[path[i,j,1], path[i,j,0], 1]
        return newpath

    def diffusion(self, w2, dt):
        scale = 10
        w2_x = w2[:,:,0].reshape(self.size)
        w2_y = w2[:,:,1].reshape(self.size)
        M, c_x, c_y = self.velocity_boundaries(w2_x, w2_y)
        L = self.I - (self.viscosity * dt)*M
        self.scale_boundaries(scale=scale)
        L1, c_x = self.scale_down(L, c_x)
        L2, c_y = self.scale_down(L, c_y)

        self.print_vector("L1", L1.todense())
        self.print_vector("L2", L2.todense())
        self.print_vector("c_x", c_x)
        self.print_vector("c_y", c_y)
        w30 = scipy.sparse.linalg.spsolve(L1, c_x)
        w31 = scipy.sparse.linalg.spsolve(L2, c_y)
        
        w30 = self.scale_up(w30)
        w31 = self.scale_up(w31)
        self.rescale_boundaries()

        w3 = np.zeros([int(self.n/self.h), int(self.m/self.h), 2])
        w3[:,:,0] = w30.reshape([int(self.n/self.h), int(self.m/self.h)])
        w3[:,:,1] = w31.reshape([int(self.n/self.h), int(self.m/self.h)])
        
        return self.reset_solid_velocities(w3)
    
    def perform_advection(self, w1, dt):
        self.path = self.update_path(self.path, w1, dt)
        w2 = self.advection(w1, dt, self.path)
        return w2

    def perform_diffusion(self, w2, dt):
        w3 = self.diffusion(w2, dt)
        return w3
