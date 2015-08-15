

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
        ax = np.linspace(0, self.m, self.m)
        ay = np.linspace(0, self.n, self.n)

        func0 = interpolate.RectBivariateSpline(ax, ay, w1[:,:,0])
        func1 = interpolate.RectBivariateSpline(ax, ay, w1[:,:,1])
        
        self.logger.print_vector("psx", psx)

        w2[:,:,0] = func0.ev(psy, psx)
        w2[:,:,1] = func1.ev(psy, psx)

        self.logger.print_vector("w2_x interpolated", w2[:,:,0])
        return w2
        
    def diffusion(self, w2, dt):
        #scale = 10
        w2_x = w2[:,:,0].reshape(self.size)
        w2_y = w2[:,:,1].reshape(self.size)
        M, c_x, c_y = self.velocity_boundaries(w2_x, w2_y)
        L = self.I - (self.viscosity * dt)*M
        #self.scale_boundaries(scale=scale)
        #L1, c_x = self.scale_down(L, c_x)
        #L2, c_y = self.scale_down(L, c_y)

        #self.logger.print_vector("L1", L1.todense())
        #self.logger.print_vector("L2", L2.todense())
        self.logger.print_vector("c_x", c_x)
        self.logger.print_vector("c_y", c_y)
        w30 = scipy.sparse.linalg.spsolve(L, c_x)
        w31 = scipy.sparse.linalg.spsolve(L, c_y)
        
        #w30 = self.scale_up(w30)
        #w31 = self.scale_up(w31)
        #self.rescale_boundaries()

        w3 = np.zeros([int(self.n/self.h), int(self.m/self.h), 2])
        w3[:,:,0] = w30.reshape([int(self.n/self.h), int(self.m/self.h)])
        w3[:,:,1] = w31.reshape([int(self.n/self.h), int(self.m/self.h)])

        return self.reset_solid_velocities(w3)
    
    def perform_advection(self, w1, dt):
        w2 = self.advection(w1, dt)
        return w2

    def perform_diffusion(self, w2, dt):
        w3 = self.diffusion(w2, dt)
        w3[:,:,0] = self.scale_up_field(self.scale_down_field(w3[:,:,0]))
        w3[:,:,1] = self.scale_up_field(self.scale_down_field(w3[:,:,1]))
        return w3
