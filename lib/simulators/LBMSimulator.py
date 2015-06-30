
from .BaseSimulator import BaseSimulator 

import numpy as np 

class LBMSimulator(BaseSimulator):
    
    def traverse_grid(self, grid, n, m, fun, **kwargs):
        """
        Traverses grid

        grid - grid to traverse on 
        n - number of rows 
        m - number of columns
        fun - function to obtain on cell, function must be in shape: fun(cell, i, j, **kwargs)
        **kwargs - additional arguments for specific function

        returns resulting array (matrix)

        NOTE: result is not numpy array but the python array type
        """
        res = [] 
        for i in range(n):
            row = [] 
            for j in range(m):
                c = fun(grid[i][j], i, j, **kwargs)
                row.append(c)
            res.append(row)
        return res

    def calculate_density(self, c, i, j, f):
        s = 0
        for a in range(9):
            s += f[i][j][a]
        density = s
        return density
    
    def calculate_velocity(self,c,i,j,f):
        u = np.array([0,0])
        for a in range(9):
            u += self.e[a] * f[i][j][a]
        velocity = u / c.density
        return velocity

    def calculate_feq(self, c, i, j):
        w = [4./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./9., 1./36., 1./36.]
        u = c.velocity
        res = np.zeros(9)
        for a in range(9):
            res = w*c.density*(1 + 3*(np.dot(self.e[a], u)) + (9./2.)*(np.dot(self.e[a], u))**2 - (3./2.)*np.dot(u, u))
        return res

    def stream_collisions(self, c, i, j, f, feq, a, omega):
        return f[i][j][a] - (f[i][j][a] - feq[i][j][a])/omega

    def start(self):
        self.n, self.m = len(self.grid), len(self.grid[0])
        self.e = [np.array([x,y]) for x in [0,1,-1] for y in [0,1,-1]]

    def finish(self):
        pass

    def step(self, old, dt):
        """
        Must return new grid object 
        """
        pass
