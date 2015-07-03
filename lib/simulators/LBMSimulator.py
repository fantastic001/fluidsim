
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
        for i in range(self.n):
            for j in range(self.m):
                for a in range(9):
                    if self.boundaries[i][j]:
                        self.f[i][j][a] = self.f[i][j][a]
                    else:
                        self.f[i][j][a] = self.f[i][j][a] - (self.f[i][j][a] - self.feq[i][j][a])/self.omega
    
    def bounce_back(self):
        noslip = [0, 2, 1, 6, 8, 7, 3, 5, 4]
        ftemp = self.f.copy()
        for i in range(self.n):
            for j in range(self.m):
                if self.boundaries[i][j]:
                    for a in range(9):
                        self.f[i][j][a] = ftemp[i][j][noslip[a]]
    
    def is_bounded(self, i, j):
        return i >= 0 and i < self.n and j >= 0 and j < self.m

    def streaming(self):
        fin = self.f.copy()
        for i in range(self.n):
            for j in range(self.m):
                for a in range(9):
                    if self.is_bounded(i + self.e[a][1], j + self.e[a][0]):
                        fin[i + self.e[a][1]][j + self.e[a][0]][a] = self.f[i][j][a]
        self.f = fin.copy()

    def start(self):
        self.n, self.m = self.densities.shape
        self.w = np.array([4./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./9., 1./36., 1./36.])
        self.feq = np.zeros([self.n,self.m,9])
        self.omega = 3*self.viscosity + 0.5
        self.e = [np.array([x,y]) for x in [0,1,-1] for y in [0,1,-1]]
        self.calculate_feq()
        self.f = self.feq.copy()

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
