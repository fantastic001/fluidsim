
from .BaseSimulator import BaseSimulator 

import numpy as np 

class LBMSimulator(BaseSimulator):
    
    def traverse_grid(self, grid, n, m, fun, *args):
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
                c = fun(grid[i][j], i, j, *args)
                row.append(c)
            res.append(row)
        return res

    def calculate_density(self, c, i, j, f):
        s = 0
        for a in range(9):
            s += f[i][j][a]
        density = s
        return density
    
    def calculate_velocity(self, grid,n, m, f):
        for i in range(n):
            for j in range(m):
                u = np.array([0, 0])
                for a in range(9):
                    u = u + self.e[a] * f[i][j][a]
                velocity = u / grid[i][j].density
                grid[i][j].velocity = velocity
        

    def calculate_feq(self, grid, n, m):
        for i in range(n):
            for j in range(m):
                u = grid[i][j].velocity
                for a in range(9):
                    cu = np.dot(self.e[a], u)
                    self.feq[i][j][a] = self.w[a]*grid[i][j].density*(1 + 3*cu + (9./2.)*(cu**2) - (3./2.)*(u[0]**2 + u[1]**2))

    def stream_collisions(self, c, i, j, f, feq, omega):
        res = np.zeros(9)
        for a in range(9):
            if c.solid:
                res[a] = f[i][j][a]
            else:
                res[a] = f[i][j][a] - (f[i][j][a] - feq[i][j][a])/omega
        return res
    
    def bounce_back(self, c, i, j, f):
        noslip = [0, 2, 1, 6, 8, 7, 3, 5, 4]
        res = np.zeros(9)
        if c.solid:
            for a in range(9):
                res[a] = f[i][j][noslip[a]]
        else:
            for a in range(9):
                res[a] = f[i][j][a]
        return res
    
    def is_bounded(self, i, j):
        return i >= 0 and i < self.n and j >= 0 and j < self.m

    def streaming(self, f):
        fin = f.copy()
        for i in range(self.n):
            for j in range(self.m):
                for a in range(9):
                    if self.is_bounded(i + self.e[a][1], j + self.e[a][0]):
                        fin[i + self.e[a][1]][j + self.e[a][0]][a] = f[i][j][a]
        return fin

    def start(self):
        self.n, self.m = len(self.grid), len(self.grid[0])
        self.w = [4./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./9., 1./36., 1./36.]
        self.feq = np.zeros([self.n,self.m,9])
        self.omega = 3*self.viscosity + 0.5
        self.e = [np.array([x,y]) for x in [0,1,-1] for y in [0,1,-1]]
        self.calculate_feq(self.grid, self.n, self.m)
        self.f = self.feq.copy()

    def finish(self):
        pass

    def step(self, old, dt):
        """
        Must return new grid object 
        """
        # IN START: initialize 
        # copy feq to fin 
        # collision step 
        new = old.copy()
        self.calculate_feq(old, self.n, self.m)
        self.f = np.array(self.traverse_grid(old, self.n, self.m, self.stream_collisions, self.f, self.feq, self.omega))
        # for every obstacle, noslip condition, reverse distribution function to opposite directions 
        self.f = np.array(self.traverse_grid(old, self.n, self.m, self.bounce_back, self.f))
        # streaming step 
        self.f = self.streaming(self.f)
        # recalculate densities and velocities 
        densities = np.array(self.traverse_grid(old, self.n, self.m, self.calculate_density, self.f))
        for i in range(self.n):
            for j in range(self.m):
                new[i][j].density = densities[i][j]
        self.calculate_velocity(new, self.n, self.m, self.f)
        return new
