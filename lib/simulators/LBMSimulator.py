
from .BaseSimulator import BaseSimulator 

import numpy as np 

class LBMSimulator(BaseSimulator):
    
    def calculate_density(self):
        p = np.zeros([self.n, self.m])
        for a in range(9):
            p += self.f[:,:,a]
        return p
    
    def calculate_velocity(self):
        u = np.zeros([self.n, self.m, 2])
        for a in range(9):
            u[:,:,0] += self.e[a][0] * self.f[:,:,a]
            u[:,:,1] += self.e[a][1] * self.f[:,:,a]
        u[:,:,0] /= self.densities 
        u[:,:,1] /= self.densities
        return u 
        

    def calculate_feq(self):
        u = self.velocities
        u2 = u[:,:,0]*u[:,:,0] + u[:,:,1]*u[:,:,1]
        one = np.ones([self.n, self.m])
        for a in range(9):
            cu = u[:,:, 0]*self.e[a][0] + u[:,:,1]*self.e[a][1]
            wp = self.w[a] * self.densities
            self.feq[:,:, a] = wp*(one + 3*cu + 4.5*cu**2 - 1.5*u2)

    def stream_collisions(self):
        for a in range(9):
            # We are multplying with non_boundaries matrix which has ones on fluid nodes and 0's on solid nodes 
            # that way we ensure that solid nodes are untouched
            self.f[:,:,a] = self.f[:,:,a] - ((self.f[:,:,a] - self.feq[:,:,a])/self.omega) * self.non_boundaries 
    
    def bounce_back(self):
        noslip = [0, 2, 1, 6, 8, 7, 3, 5, 4]
        ftemp = self.f.copy()
        for a in range(9):
            self.f[:,:,a] = ftemp[:,:,noslip[a]]*self.boundaries + self.f[:,:,a]*self.non_boundaries
    
    def is_bounded(self, i, j):
        return i >= 0 and i < self.n and j >= 0 and j < self.m

    def streaming(self):
        fin = self.f.copy()
        for a in range(1, 9):
            fin[:,:,a] = np.roll(self.f[:,:,a], -self.e[a][1], axis=0)
            fin[:,:,a] = np.roll(fin[:,:,a], self.e[a][0], axis=1)
        # Restore ruined 
        #fin[-1,:,1] = self.f[-1,:,1]
        #fin[0,:,2] = self.f[0,:,2]
        #fin[:,0,3] = self.f[:,0,3]
        #fin[-1,:,4] = self.f[-1,:,4]
        #fin[:,0,4] = self.f[:,0,4]
        #fin[0,:,5] = self.f[0,:,5]
        #fin[:,0,5] = self.f[:,0,5]
        #fin[:,-1,6] = self.f[:,-1,6]
        #fin[-1,:,7] = self.f[-1,:,7]
        #fin[:,-1,7] = self.f[:,-1,7]
        #fin[0,:,8] = self.f[0,:,8]
        #fin[:,-1,8] = self.f[:,-1,8]
        self.f = fin.copy()

    def start(self):
        self.n, self.m = self.densities.shape
        self.w = np.array([4./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./9., 1./36., 1./36.])
        self.feq = np.zeros([self.n,self.m,9])
        self.omega = 3*self.viscosity + 0.5
        self.e = [np.array([x,y]) for x in [0,1,-1] for y in [0,1,-1]]
        self.calculate_feq()
        self.f = self.feq.copy()
        self.non_boundaries = np.ones(self.boundaries.shape) - self.boundaries

    def finish(self):
        pass

    def step(self, dt):
        """
        Must return new grid object 
        """
        # IN START: initialize 
        # copy feq to fin 
        # collision step 
        if np.isnan(self.feq).any() or np.isnan(self.f).any():
            print("WARNING NAN FOUND")
        self.calculate_feq()
        self.stream_collisions()
        # for every obstacle, noslip condition, reverse distribution function to opposite directions 
        self.bounce_back()
        # streaming step 
        self.streaming()
        # recalculate densities and velocities 
        self.densities = self.calculate_density()
        self.velocities = self.calculate_velocity()
