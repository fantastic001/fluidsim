

import numpy as np
import numpy.linalg 

from .CFDSimulator import *

import scipy.interpolate as interpolate
import scipy.sparse
import scipy.sparse.linalg

import os.path

class CFDImplicitSimulator(CFDSimulator):

    def advection(self, w1, dt):
        w2 = w1.copy()
        
        psy = self.y - dt*w1[:, :, 1]
        psx = self.x - dt*w1[:, :, 0]
        ax = np.linspace(0, self.m-1, self.m)
        ay = np.linspace(0, self.n-1, self.n)

        func0 = interpolate.RectBivariateSpline(ax, ay, w1[:,:,0], kx=1, ky=1)
        func1 = interpolate.RectBivariateSpline(ax, ay, w1[:,:,1], kx=1, ky=1)
        
        self.logger.print_vector("psx", psx)

        w2[:,:,0] = func0.ev(psy, psx)
        w2[:,:,1] = func1.ev(psy, psx)

        self.logger.print_vector("w2_x interpolated", w2[:,:,0])
        return w2
        
    def diffusion(self, w2, dt):
        #scale = 10
        w2_x = w2[:,:,0].reshape(self.size)
        w2_y = w2[:,:,1].reshape(self.size)
        
        #self.scale_boundaries(scale=scale)
        #L1, c_x = self.scale_down(L, c_x)
        #L2, c_y = self.scale_down(L, c_y)

        w30 = None
        w31 = None
        if dt == 0.1:
            w30 = self.vsolver.solve(w2_x)
            w31 = self.vsolver.solve(w2_y)
        else:
            M, c_x, c_y = self.velocity_boundaries(w2_x, w2_y)
            L = self.I - (self.viscosity * dt)*self.Av

            self.logger.print_vector("c_x", c_x)
            self.logger.print_vector("c_y", c_y)
            
            w30 = scipy.sparse.linalg.spsolve(L, w2_x)
            w31 = scipy.sparse.linalg.spsolve(L, w2_y)
        
        #w30 = self.scale_up(w30)
        #w31 = self.scale_up(w31)
        #self.rescale_boundaries()

        w3 = np.zeros([int(self.n/self.h), int(self.m/self.h), 2])
        w3[:,:,0] = w30.reshape([int(self.n/self.h), int(self.m/self.h)])
        w3[:,:,1] = w31.reshape([int(self.n/self.h), int(self.m/self.h)])
        w3 = np.around(w3, 6)
        return w3
    
    def perform_advection(self, w1, dt):
        w2 = self.advection(w1, dt)
        return w2

    def perform_diffusion(self, w2, dt):
        w3 = self.diffusion(w2, dt)
        return w3
