
import numpy as np
import numpy.linalg 

from .BaseSimulator import BaseSimulator

import scipy.interpolate as interpolate
import scipy.sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt 

class CFDSimulator(BaseSimulator):
    
    DEBUG = True
    DEBUG_BREAK = False
    DEBUG_PLOT = False

    def compute_gradient(self, a, dx, dy):
        dy, dx = np.gradient(a, dy, dx, edge_order=2)
        grad = np.zeros([a.shape[0], a.shape[1], 2])
        grad[:,:,0] = dx 
        grad[:,:,1] = dy
        return grad

    def compute_divergence(self, f, dx, dy):
        p = f[:,:,0]
        q = f[:,:,1]
        dp = self.compute_gradient(p, dx, dy) 
        dq = self.compute_gradient(q, dx, dy)
        dpdx = dp[:,:,0]
        dqdy = dq[:,:,1]
        return dpdx + dqdy
    
    def compute_laplacian(self, f, dx, dy):
        p = f[:,:,0]
        q = f[:,:,1]
        dp = self.compute_gradient(p, dx, dy)
        dq = self.compute_gradient(q, dx, dy)
        dpdx = dp[:,:,0]
        dpdy = dp[:,:,1]
        dqdx = dq[:,:,0]
        dqdy = dq[:,:,1]

        dpxx = self.compute_gradient(dpdx, dx, dy)[:,:,0]
        dpyy = self.compute_gradient(dpdy, dx, dy)[:,:,1]
        dqxx = self.compute_gradient(dqdx, dx, dy)[:,:,0]
        dqyy = self.compute_gradient(dqdy, dx, dy)[:,:,1]
        
        res = f.copy()
        res[:,:,0] = dpxx + dpyy
        res[:,:,1] = dqxx + dqyy
        return res 
    
    def boundary_up(self, i,j):
        return i==0 or self.boundaries[i-1, j]
    def boundary_down(self, i,j):
        return i==int(self.n/self.h)-1 or self.boundaries[i+1, j]
    def boundary_left(self, i,j):
        return j==0 or self.boundaries[i, j-1]
    def boundary_right(self, i,j):
        return j==int(self.m/self.h)-1 or self.boundaries[i, j+1]

    def boundary(self, i,j):
        return self.boundary_up(i,j) or self.boundary_left(i,j) or self.boundary_right(i,j) or self.boundary_down(i,j)

    def get_laplacian_operator(self):
        # Computing laplacian operator 
        size = int(self.n/self.h) * int(self.m / self.h)
        c = int(self.m/self.h)
        A = scipy.sparse.dok_matrix((size, size))
        for iy1 in self.ay:
            for ix1 in self.ax:
                iix1 = int(ix1/self.h)
                iiy1 = int(iy1/self.h)
                s = (self.m * iiy1)/self.h + iix1
                #print("ix iy = %d %d" % (ix1, iy1))
                #print("iix1, iiy1 = %d %d" % (iix1, iiy1))
                #print("index %d" % s)
                if not self.boundary(iiy1, iix1):
                    A[s,s] = -4
                    A[s, s+1] = 1
                    A[s, s-1] = 1
                    A[s,s+c] = 1 
                    A[s,s-c] = 1
        I = scipy.sparse.eye(size)
        return (A, I, size)

    def pressure_boundaries(self, M, c):
        A = M.copy()
        b = c.copy()
        columns = int(self.m/self.h)
        rows = int(self.n/self.h)
        for i in range(rows):
            for j in range(columns):
                s = columns * i + j
                if self.boundary(i,j):
                    A[s,s] = 1
                    b[s] = 0
                    if self.boundary_up(i,j):
                        A[s,s+columns] = -1
                    if self.boundary_down(i,j):
                        A[s,s-columns] = -1
                    if self.boundary_left(i,j):
                        A[s,s+1] = -1
                    if self.boundary_right(i,j):
                        A[s,s-1] = -1
        return (A,b)
    
    def velocity_boundaries(self, M, c_x, c_y):
        A = M.copy()
        b_x = c_x.copy()
        b_y = c_y.copy()
        columns = int(self.m/self.h)
        rows = int(self.n/self.h)
        for i in range(rows):
            for j in range(columns):
                s = columns * i + j
                if self.boundary(i,j):
                    A[s,s] = 1
                    if self.boundary_up(i,j) or self.boundary_down(i,j):
                        b_y[s] = 0
                    if self.boundary_right(i,j) or self.boundary_left(i,j):
                        b_x[s] = 0
        return (A,b_x, b_y)
    
    def start(self):
        self.deltas = []

        self.h = 1.0
        self.n, self.m = (self.velocities.shape[0]*self.h, self.velocities.shape[1]*self.h)
        self.forces = np.zeros([int(self.n/self.h), int(self.m/self.h), 2]) # Will be removed and modeled differently
        # Set forces :)))
        for fi in range(int(self.n/self.h)):
            self.forces[fi, :, 0] = np.linspace(100, 0, self.m/self.h)
        
        self.y,self.x = np.mgrid[0:self.n:self.h, 0:self.m:self.h]
        self.ax = np.arange(0, self.m, self.h)
        self.ay = np.arange(0, self.n, self.h)

        # set path 
        self.path = np.zeros([int(self.n / self.h), int(self.m / self.h), 2])
        for py in self.ay:
            for px in self.ax:
                self.path[int(py/self.h), int(px/self.h), 0] = px
                self.path[int(py/self.h), int(px/self.h), 1] = py
        self.A, self.I, self.size = self.get_laplacian_operator()
        self.bmap = np.zeros([int(self.n/self.h), int(self.m/self.h)])
        self.iteration = 0
    
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
        s[:,:,0] = u_x * duxdx + u_y * duydx
        s[:,:,1] = u_x * duxdy + u_y * duydy
        return s

    def add_force(self, v, dt, f):
        return v + dt*f

    def advection(self, v, dt, path):
        w2 = v.copy()
        for j in self.ax:
            for i in self.ay:
                psi = path[i,j,1]
                psj = path[i,j,0]
                psi = np.clip(int(psi), 0, self.n/self.h - 1)
                psj = np.clip(int(psj), 0, self.m/self.h - 1)
                w2[i,j,0] = w1[psi, psj, 0]
                w2[i,j,1] = w1[psi, psj, 1]
        return w2
        
    def update_path(self, path, w1, dt):
        # set new paths 
        newpath = path.copy()
        newpath[:,:,0] = path[:,:,0] + w1[:,:,0]*dt
        newpath[:,:,1] = path[:,:,1] + w1[:,:,1]*dt
        return newpath

    def diffusion(self, w2, dt):
        w2_x = w2[:,:,0].reshape(self.size)
        w2_y = w2[:,:,1].reshape(self.size)
        M, c_x, c_y = self.velocity_boundaries(self.A, w2_x, w2_y)
        w30 = scipy.sparse.linalg.spsolve(self.I - (self.viscosity * dt)*M, c_x)
        w31 = scipy.sparse.linalg.spsolve(self.I - (self.viscosity * dt)*M, c_y)
        w3 = np.zeros([int(self.n/self.h), int(self.m/self.h), 2])
        w3[:,:,0] = w30.reshape([int(self.n/self.h), int(self.m/self.h)])
        w3[:,:,1] = w31.reshape([int(self.n/self.h), int(self.m/self.h)])
        return w3

    def projection(self, w3, dt):
        div_w3 = self.compute_divergence(w3, self.h, self.h)
        div_w3_reshaped = div_w3.reshape(self.size)
        M, c = self.pressure_boundaries(self.A, div_w3_reshaped)
        p_ = scipy.sparse.linalg.spsolve(M, c)
        p = p_.reshape(self.n/self.h, self.m/self.h)
        grad_p = self.compute_gradient(p, self.h, self.h)
        w4 = w3 - grad_p 
        return (w4, p)


    def compute_speed(self, v):
        return np.sqrt(v[:, :, 0]**2 + v[:, :, 1]**2)

    def finish(self):
        #plt.plot(np.diff(self.deltas))
        #plt.show()
        pass
    
    def plot_change(self, iteration, v, bmap):
        if self.DEBUG_PLOT:
            plt.imsave("debug-%d.png" % iteration, self.compute_speed(v) - bmap, vmax=2, vmin=-2)
            iteration += 1
            bmap = self.compute_speed(v)
        return (bmap, iteration)
    
    def print_vector(self, s, v):
        if self.DEBUG:
            print(s)
            print(v)
            if self.DEBUG_BREAK:
                input()

    def step(self, dt):
        w0 = self.velocities 
        self.print_vector("w0", w0[:,:,0])
        self.bmap, self.iteration = self.plot_change(self.iteration, w0, self.bmap)
        
        w1 = self.add_force(w0, dt, self.forces)
        self.print_vector("w1", w1[:,:,0])
        self.bmap, self.iteration = self.plot_change(self.iteration, w1, self.bmap)
        
        # w2 = self.advection(w1, dt, self.path)
        # self.path = self.update_path(self.path, w1, dt)
        w2 = w1.copy()
        w2 =  w1 - dt * self.advection_primitive(w1)
        self.print_vector("w2", w2[:,:,0])
        self.bmap, self.iteration = self.plot_change(self.iteration, w2, self.bmap)
        
        # w3 = self.diffusion(w2, dt)
        w3 = w2 + dt * self.viscosity * self.compute_laplacian(w2, self.h, self.h)
        self.print_vector("w3", w3[:,:,0])
        self.bmap, self.iteration = self.plot_change(self.iteration, w3, self.bmap)
        
        w4, p = self.projection(w3, dt)
        self.velocities = w4 
        self.pressure = p 

        self.densities = p 
        
        self.print_vector("div v", self.compute_divergence(self.velocities, self.h, self.h))
        self.bmap, self.iteration = self.plot_change(self.iteration, w4, self.bmap)
        
        self.forces.fill(0)
