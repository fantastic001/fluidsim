
import numpy as np
import numpy.linalg 

from .BaseSimulator import BaseSimulator

import scipy.interpolate as interpolate
import scipy.sparse
import scipy.sparse.linalg

import os.path

from ..utils import staggered

class CFDSimulator(BaseSimulator):

    def compute_speed(self, v):
        return np.sqrt(v[:, :, 0]**2 + v[:, :, 1]**2)
    
    def compute_gradient(self, a, dx, dy, edge_order=2):
        dy, dx = np.gradient(a, dy, dx, edge_order=edge_order)
        grad = np.zeros([a.shape[0], a.shape[1], 2])
        grad[:,:,0] = dx 
        grad[:,:,1] = dy
        return grad

    def compute_divergence(self, f, dx, dy, edge_order=2):
        p = f[:,:,0]
        q = f[:,:,1]
        dp = self.compute_gradient(p, dx, dy, edge_order=edge_order) 
        dq = self.compute_gradient(q, dx, dy, edge_order=edge_order)
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
        return i==self.boundaries.shape[0]-1 or self.boundaries[i+1, j]
    def boundary_left(self, i,j):
        return j==0 or self.boundaries[i, j-1]
    def boundary_right(self, i,j):
        return j==self.boundaries.shape[1]-1 or self.boundaries[i, j+1]
    
    def boundary_edge(self, i, j):
        return (self.boundary_up(i,j) and self.boundary_left(i,j)) or (self.boundary_up(i,j) and self.boundary_right(i,j)) or (self.boundary_down(i,j) and self.boundary_left(i,j)) or (self.boundary_down(i,j) and self.boundary_right(i,j))

    def boundary(self, i,j):
        return self.boundary_up(i,j) or self.boundary_left(i,j) or self.boundary_right(i,j) or self.boundary_down(i,j)

    def get_laplacian_operator(self):
        # Computing laplacian operator 
        size = int(self.n/self.h) * int(self.m / self.h)
        c = int(self.m/self.h)
        I = scipy.sparse.eye(size)

        A = scipy.sparse.lil_matrix((size, size))
        for iiy1 in range(int(self.n)):
            for iix1 in range(int(self.m)):
                s = (self.m * iiy1) + iix1
                if not self.boundary(iiy1, iix1):
                    A[s,s] = -4
                    A[s, s+1] = 1
                    A[s, s-1] = 1
                    A[s,s+c] = 1 
                    A[s,s-c] = 1
        return (A, I, size)
    
    def get_pressure_laplacian_operator(self, scale=10):
        size = scale**2
        b = np.ones(size)
        A = np.zeros([size, size])
        columns = scale
        rows = scale
        for i in range(rows):
            for j in range(columns):
                s = columns * i + j
                if self.boundaries[i,j]:
                    A[s,s] = -4
                    A[s,s-1] = 1
                    A[s,s+1] = 1
                    A[s,s+columns] = 1
                    A[s,s-columns] = 1
                    b[s] = 0
                else:
                    if self.boundary(i,j):
                        A[s,s] = -2
                        if self.boundary_edge(i,j):
                            A[s,s] = 1 
                            b[s] = 0
                            continue
                        if self.boundary_up(i,j):
                            A[s,s+1] = 1
                            A[s,s-1] = 1
                        if self.boundary_down(i,j):
                            A[s,s+1] = 1
                            A[s,s-1] = 1
                        if self.boundary_left(i,j):
                            A[s,s+columns] = 1
                            A[s,s-columns] = 1
                        if self.boundary_right(i,j):
                            A[s,s + columns] = 1
                            A[s,s - columns] = 1
                    else:
                        A[s,s] = -4
                        A[s,s+1] = 1
                        A[s,s-1] = 1 
                        A[s,s+columns] = 1 
                        A[s,s-columns] = 1
        return (A, b)

    def pressure_boundaries(self, c):
        Ap, bp = self.get_pressure_laplacian_operator()
        b = c*bp
        return (Ap,b)
    
    def get_velocity_laplacian_operator(self, M):
        A = M.copy()
        b_x = np.ones(self.size)
        b_y = np.ones(self.size)
        columns = int(self.m/self.h)
        rows = int(self.n/self.h)
        for i in range(rows):
            for j in range(columns):
                s = columns * i + j
                if self.boundaries[i,j]:
                    A[s,s] = -4
                    A[s,s+columns] = 1
                    A[s,s-columns] = 1
                    A[s,s+1] = 1 
                    A[s,s-1] = 1
                if self.boundary(i,j):
                    A[s,s] = -4
                    if not self.boundary_down(i,j):
                        A[s,s+columns] = 1
                    if not self.boundary_up(i,j):
                        A[s,s-columns] = 1
                    if not self.boundary_right(i,j):
                        A[s,s+1] = 1 
                    if not self.boundary_left(i,j):
                        A[s,s-1] = 1
        return (A,b_x, b_y)

    def velocity_boundaries(self, c_x, c_y):
        return (self.Av, c_x*self.bv_x, c_y*self.bv_y)
    
    def reset_solid_velocities(self, v):
        w = v.copy()
        n,m,d = v.shape
        for i in range(n):
            for j in range(m):
                if self.boundaries[i,j]:
                    w[i,j] = 0
        return w 
    
    def add_force(self, v, dt, f):
        return v + dt*f
    
    def projection(self, w3, dt):
        n,m,d = w3.shape
        # Converting to staggered 
        u,v = staggered.field_transpose(w3[:,:,0], w3[:,:,1]) 
        u,v = staggered.to_staggered(u,v)

        u,v = staggered.reset_solids(u,v, self.boundaries.T)
        u,v, p = staggered.projection(u,v, self.psolver, self.boundaries.T)
        
        u,v = staggered.reset_solids(u,v, self.boundaries.T)
        
        print("Divergence error")
        ubc, vbc = staggered.attach_boundaries(u,v)
        div = staggered.compute_divergence(ubc, vbc)
        print(np.abs(div).max())

        # bring 'em back 
        #ubc, vbc = staggered.attach_boundaries(u,v)
        ubc,vbc = u,v 

        u, v = staggered.to_centered(ubc, vbc)
        u,v = staggered.field_transpose(u,v)
        w4 = w3.copy()
        w4[:,:,0] = u 
        w4[:,:,1] = v
        return (w4, p)
        
    def start(self):
        print("Computing psolver")
        self.psolver = staggered.set_solids(self.boundaries.T)
        print("psolver computed")

        self.h = 1.0
        self.n, self.m = (self.velocities.shape[0]*self.h, self.velocities.shape[1]*self.h)
        
        self.non_boundaries = np.ones([int(self.n), int(self.m)]) - self.boundaries
        
        self.pressure = np.zeros([int(self.n/self.h), int(self.m/self.h)])

        self.y, self.x = np.mgrid[0:int(self.n), 0:int(self.m)]
        self.A, self.I, self.size = self.get_laplacian_operator()
        self.bmap = np.zeros([int(self.n/self.h), int(self.m/self.h)])
        self.iteration = 0
        
        self.Av, self.bv_x, self.bv_y = self.get_velocity_laplacian_operator(self.A)
        L = self.I - (self.viscosity * 0.1)*self.Av
        print("Computing vsolver")
        self.vsolver = scipy.sparse.linalg.splu(L)
        print("vsolver computed")
    
    def finish(self):
        #plt.plot(np.diff(self.deltas))
        #plt.show()
        pass
    
    def perform_advection(self, w1, dt):
        pass

    def perform_diffusion(self, w2, dt):
        pass

    def perform_projection(self, w3, dt):
        w4, p = self.projection(w3, dt)
        return (w4, p)

    def step(self, dt):
        w0 = self.velocities 
        self.logger.print_vector("w0 x", w0[:,:,0])
        self.logger.print_vector("w0 y", w0[:,:,1])
        self.logger.print_vector("div w0", self.compute_divergence(w0, self.h, self.h))
        self.logger.plot_field(w0, "%d-w0" % self.iteration)
        self.logger.print_vector("w0 x error:", np.abs(w0[:,:,0]).max())
        self.logger.print_vector("w0 y error:", np.abs(w0[:,:,1]).max())

        w1 = self.add_force(w0, dt, self.forces)
        self.logger.print_vector("w1 x", w1[:,:,0])
        self.logger.print_vector("w1 y", w1[:,:,1])
        self.logger.print_vector("div w1", self.compute_divergence(w1, self.h, self.h))
        self.logger.plot_field(w1, "%d-w1" % self.iteration)
        self.logger.print_vector("w1 x error:", np.abs(w1[:,:,0]).max())
        self.logger.print_vector("w1 y error:", np.abs(w1[:,:,1]).max())
        
        w2 = self.perform_advection(w1, dt)
        self.logger.print_vector("w2 x", w2[:,:,0])
        self.logger.print_vector("w2 y", w2[:,:,1])
        self.logger.print_vector("div w2", self.compute_divergence(w2, self.h, self.h))
        self.logger.plot_field(w2, "%d-w2" % self.iteration)
        self.logger.print_vector("w2 x error:", np.abs(w2[:,:,0]).max())
        self.logger.print_vector("w2 y error:", np.abs(w2[:,:,1]).max())
        
        w3 = self.perform_diffusion(w2, dt)
        self.logger.print_vector("w3 x", w3[:,:,0])
        self.logger.print_vector("w3 y", w3[:,:,1])
        self.logger.print_vector("div w3", self.compute_divergence(w3, self.h, self.h))
        self.logger.plot_field(w3, "%d-w3" % self.iteration)
        self.logger.print_vector("w3 x error:", np.abs(w3[:,:,0]).max())
        self.logger.print_vector("w3 y error:", np.abs(w3[:,:,1]).max())

        w4, p = self.perform_projection(w3, dt)
        self.logger.print_vector("w4 x", w4[:,:,0])
        self.logger.print_vector("w4 y", w4[:,:,1])
        self.logger.print_vector("div w4", self.compute_divergence(w4, self.h, self.h))
        self.velocities = self.reset_solid_velocities(w4)
        self.pressure = p 
        self.logger.plot_field(w4, "%d-w4" % self.iteration)
        self.logger.print_vector("w4 x error:", np.abs(w4[:,:,0]).max())
        self.logger.print_vector("w4 y error:", np.abs(w4[:,:,1]).max())
        
        self.logger.print_vector("div v", self.compute_divergence(self.velocities, self.h, self.h))
        
        self.logger.print_vector("divergence error", np.abs(self.compute_divergence(self.velocities, self.h, self.h)).max())

        self.logger.print_vector("Substance sum: ", self.densities.sum())
        self.logger.print_vector("Substance: ", self.densities)
        self.iteration += 1
