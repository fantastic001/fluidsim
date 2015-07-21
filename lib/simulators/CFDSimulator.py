
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

    def is_edge(self, i,j):
        return [i == 0, j == self.m/self.h - 1, i == self.n/self.h - 1, j == 0]

    def is_out(self,i,j):
        return i < 0 or i == self.n/self.h or j < 0 or j == self.m/self.h 
    
    def get_laplacian_operator(self):
        # Computing laplacian operator 
        size = int(self.n/self.h) * int(self.m / self.h)
        A = scipy.sparse.dok_matrix((size, size))
        ne_conditions = [np.array([0,1]), np.array([0,-1]), np.array([1, 0]), np.array([-1, 0])]
        #print("ax = %d" % len(self.ax))
        for iy1 in self.ay:
            for ix1 in self.ax:
                iix1 = int(ix1/self.h)
                iiy1 = int(iy1/self.h)
                s = (self.m * iiy1)/self.h + iix1
                #print("ix iy = %d %d" % (ix1, iy1))
                #print("iix1, iiy1 = %d %d" % (iix1, iiy1))
                #print("index %d" % s)
                A[s,s] = -4
                edges = self.is_edge(iiy1, iix1)
                for edge in edges:
                    if edge:
                        A[s, s] = A[s,s] + 1
                for condition in ne_conditions:
                    iiy2 = iiy1 + condition[1]
                    iix2 = iix1 + condition[0]
                    if not self.is_out(iiy2, iix2):
                        s2 = (self.m * iiy2)/self.h + iix2
                        A[s, s2] = 1
                        A[s2, s] = 1
                        #if s == 1:
                        #    print("CONDITION %d %d" % (s, s2))
        I = scipy.sparse.eye(size)
        return (A, I, size)

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
        self.path = np.zeros([self.n / self.h, self.m / self.h, 2])
        for py in self.ay:
            for px in self.ax:
                self.path[py/self.h, px/self.h, 0] = px
                self.path[py/self.h, px/self.h, 1] = py
        self.A, self.I, self.size = self.get_laplacian_operator()
        self.bmap = np.zeros([self.n/self.h, self.m/self.h])
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
        w30, info = scipy.sparse.linalg.bicg(self.I - (self.viscosity * dt)*self.A, w2_x)
        w31, info = scipy.sparse.linalg.bicg(self.I - (self.viscosity * dt)*self.A, w2_y)
        w3 = np.zeros([self.n/self.h, self.m/self.h, 2])
        w3[:,:,0] = w30.reshape([self.n/self.h, self.m/self.h])
        w3[:,:,1] = w31.reshape([self.n/self.h, self.m/self.h])
        return w3

    def projection(self, w3, dt):
        div_w3 = self.compute_divergence(w3, self.h, self.h)
        div_w3_reshaped = div_w3.reshape(self.size)
        
        p_ = scipy.sparse.linalg.spsolve(self.A, div_w3_reshaped)
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
