
from .BaseSimulator import BaseSimulator 

import numpy as np 

class LBMSimulator(BaseSimulator):
    
    def calculate_densities(self, f):
        for i in range(self.n):
            for j in range(self.m):
                s = 0
                for a in range(9):
                    s += f[i][j][a]
                self.grid[i][j].density = s
    
    def calculate_velocities(self):
        for i in range(self.n):
            for j in range(self.m):
                u = np.array([0,0])
                for a in range(9):
                    u += self.e[a] * f[i][j][a]
                self.grid[i][j].velocity = u / self.grid[i][j].density

    def start(self):
        self.n, self.m = self.grid.shape
        self.e = [np.array([x,y]) for x in [0,1,-1] for y in [0,1,-1]]


    def finish(self):
        pass

    def step(self):
        pass

