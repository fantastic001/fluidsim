
import numpy as np

from .BaseSimulator import BaseSimulator

import scipy.interpolate as interpolate
import scipy.sparse
import scipy.sparse.linalg

class CFDSimulator(BaseSimulator):

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

    def start(self):
        self.h = 0.001
        self.old = self.velocities
        try:
            self.forces 
        except NameError:
            self.forces = np.zeros([self.n, self.m])
    
    def finish(self):
        pass

    def step(self, dt):
        # Setting some stuff for later computation
        #self.old = self.velocities.copy()
        y,x = np.mgrid[0:100:h, 0:100:h]
        ax = np.arange(0, self.m, h)
        ay = np.arange(0, self.n, h)

        w0 = self.velocities 
        w1 = w0 + dt * self.forces 
        w2 = w1.copy()
        # compute velocity at past 
        w1p0 = interpolate.RectBivariateSpline(ax, ay, self.old[:,:,0])
        w1p1 = interpolate.RectBivariateSpline(ax, ay, self.old[:,:,1])
        for i in ax:
            for j in ay:
                i_ = i - dt*w1[i,j, 0]
                j_ = j - dt*w1[i,j, 1]
                
                # we do not want to go outside of the boundary 
                i_ = np.clip(i_, 0, self.m)
                j_ = np.clip(i_, 0, self.n)
                w2[i,j,0] = w1p0(i_, j_)
                w2[i,j,1] = w1p1(i_, j_)
        size = self.n * self.m 
        
        # calculating diffusion 
        A = scipy.sparse.dok_matrix((size, size))
        for ix in ax:
            for iy in ay:
                s = self.m * iy + ix 
                s1 = self.m*(iy + h) + ix 
                s2 = self.m*iy + ix+h
                s3 = self.m*(iy - h) + ix 
                s4 = self.m*iy + ix - h
                A[s][s] = 1 - 4/(h**2)
                
                # s1
                if iy + h == self.n-1:
                    A[s][s1] = 0 
                else:
                    A[a][s1] = 1/h**2

                # s2 
                if ix + h == self.m - 1:
                    A[s][s2] = 0
                else:
                    A[s][s2] = 1/h**2

                # s3 
                if iy - h == 0:
                    A[s][s3] = 0
                else:
                    A[s][s3] = 1/h**2

                # s4 
                if ix - h == 0:
                    A[s][a4] = 0
                else:
                    A[s][s4] = 1/h**2
        
        # calculating w3 
        w2_x = w2[:,:,0].reshape(size)
        w2_y = w2[:,:,1].reshape(size)
        w30 = scipy.sparse.linalg.spsolve(A, w2_x)
        w31 = scipy.sparse.linalg.spsolve(A, w2_y)

        w3 = np.zeros([self.n, self.m, 2])
        w3[:,:,0] = w30.reshape([self.n, self.m])
        w3[:,:,1] = w31.reshape([self.n, self.m])

        # OH YEAH ! 
        # We have w3, now we can compute pressure, finally 
        # Qp = div_w3
        div_w3 = self.compute_divergence(w3, h, h)
        div_w3_reshaped = div_w3.reshape(size)
        Q = scipy.sparse.dok_matrix((size, size))
        for ix in ax:
            for iy in ay:
                s = self.m * iy + ix 
                s1 = self.m*(iy + h) + ix 
                s2 = self.m*iy + ix+h
                s3 = self.m*(iy - h) + ix 
                s4 = self.m*iy + ix - h
                #Q[s][s] = -4/(h**2)
                if (ix, iy) in [(0, 0), (0, self.n), (self.m, 0), (self.m, self.n)]:
                    Q[s][s] = 0
                elif ix + h == self.m or self.boundaries[iy, ix + h] or ix - h < 0 or self.boundaries[iy, ix-h]:
                    Q[s][s] = -2/(h**2)
                    Q[s][s1] = 1/h**2
                    Q[s][s3] = 1/h**2
                elif iy + h == self.n or self.boundaries[iy + h, ix] or iy - h < 0 or self.boundaries[iy-h, ix]:
                    Q[s][s2] = 1/h**2
                    Q[s][s4] = 1/h**2
                    Q[s][s] = -2/(h**2)
                else:
                    Q[s][s] = -4/(h**2)
                    Q[s][s1] = 1/h**2 
                    Q[s][s2] = 1/h**2
                    Q[s][s3] = 1/h**2
                    Q[s][s4] = 1/h**2
        p_ = scipy.sparse.linalg.spsolve(Q, div_w3_reshaped)
        p = p_.reshaped(self.n, self.m)
        grad_p = self.compute_gradient(p, h, h)
        w4 = w3 - grad_p 
        self.velocities = w4 

        # copy w1 to old 
        self.old = w1.copy()
        self.pressure = p 

        # This is wrong but i use it because i can and it is easy to debug with existing animators 
        self.densities = p 
