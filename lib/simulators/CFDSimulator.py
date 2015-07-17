
import numpy as np
import numpy.linalg 

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

    def is_edge(self, i,j):
        return [i == 0, j == self.m, i == self.n, j == 0]

    def is_out(self,i,j):
        return i < 0 or i == self.n or j < 0 or j == self.m

    def start(self):
        self.h = 0.1
        self.n, self.m = (self.velocities.shape[0]*self.h, self.velocities.shape[1]*self.h)
        self.forces = np.zeros([int(self.n/self.h), int(self.m/self.h), 2]) # Will be removed and modeled differently
        self.old = self.velocities.copy()

    def finish(self):
        pass

    def step(self, dt):
        # Setting some stuff for later computation
        #self.old = self.velocities.copy()
        h = self.h
        y,x = np.mgrid[0:self.n:h, 0:self.m:h]
        ax = np.arange(0, self.m, h)
        ay = np.arange(0, self.n, h)

        w0 = self.velocities 
        w1 = w0 + dt * self.forces 
        w2 = w1.copy()
        # compute velocity at past 
        w1p0 = interpolate.RectBivariateSpline(ay, ax, self.old[:,:,0])
        w1p1 = interpolate.RectBivariateSpline(ay, ax, self.old[:,:,1])
        for j in ax:
            for i in ay:
                i_ = i - dt*w1[i,j, 0]
                j_ = j - dt*w1[i,j, 1]
                
                # we do not want to go outside of the boundary 
                i_ = np.clip(i_, 0, self.m)
                j_ = np.clip(i_, 0, self.n)
                
                w2[i,j,0] = w1p0(i_, j_)
                w2[i,j,1] = w1p1(i_, j_)
        size = int(self.n/h) * int(self.m / h)
        
        # calculating diffusion 
        A = scipy.sparse.csc_matrix((size, size))
        """
        for iy in ay:
            for ix in ax:
                s = self.m * iy/h + ix/h 
                s1 = self.m*(iy/h + 1) + ix/h
                s2 = self.m*iy/h + (ix/h)+1
                s3 = self.m*(iy/h - 1) + ix/h
                s4 = self.m*iy/h + (ix/h) - 1
                s = int(s)
                s1 = int(s1)
                s2 = int(s2)
                s3 = int(s3) 
                s4 = int(s4)

                iix, iiy = (int(ix/h), int(iy/h))
                if (ix, iy) in [(0,0), (0, self.n-1), (self.m-1, 0), (self.m-1, self.n-1)]:
                    A[s,s] = 1 + 2/(h**2)
                    if (ix, iy) == (0,0):
                        A[s, s2] = -1/(h**2)
                        A[s, s3] = -1/(h**2)
                    if (ix, iy) == (0,self.m-1):
                        A[s, s4] = -1/(h**2)
                        A[s, s3] = -1/(h**2)
                    if (ix, iy) == (self.n-1,0):
                        A[s, s2] = -1/(h**2)
                        A[s, s1] = -1/(h**2)
                    if (ix, iy) == (self.n-1, self.m-1):
                        A[s, s1] = -1/(h**2)
                        A[s, s4] = -1/(h**2)
                elif iy == 0:
                    A[s, s2] = -1/h**2
                    A[s, s3] = -1/h**2
                    A[s, s4] = -1/h**2
                    A[s, s] = 1 + 3/(dt * self.viscosity*h**2)
                elif iy == self.n-1:
                    A[s, s1] = -1/h**2
                    A[s, s2] = -1/h**2
                    A[s, s4] = -1/h**2
                    A[s, s] = 1 + 3/(dt*self.viscosity * h**2)
                elif ix == 0:
                    A[s, s1] = -1/h**2
                    A[s, s2] = -1/h**2
                    A[s, s3] = -1/h**2
                    A[s, s] = 1 + 3/(dt*self.viscosity * h**2)
                elif ix == self.m-1:
                    A[s, s1] = -1/h**2
                    A[s, s3] = -1/h**2
                    A[s, s4] = -1/h**2
                    A[s, s] = 1 + 3/(dt*self.viscosity * h**2)
                #elif ix == 0 or ix == self.m-1 or self.boundaries[iiy, iix]:
                #    A[s, s1] = -1/h**2
                #    A[s, s3] = -1/h**2
                #    A[s,s] = 1 + 2/(dt* self.viscosity * h**2)
                else:
                    A[s, s1] = -1/h**2
                    A[s, s2] = -1/h**2
                    A[s, s3] = -1/h**2
                    A[s, s4] = -1/h**2
                    A[s, s] = 1 + 4/(dt * self.viscosity *h**2)
        """
        ne_conditions = [np.array([0,1]), np.array([0,-1]), np.array([1, 0]), np.array([-1, 0])]
        for ix1 in ax:
            for iy1 in ay:
                iix1 = int(ix1/h)
                iiy1 = int(iy1/h)
                s = (self.m * iiy1) + iix1
                A[s,s] = -4
                edges = self.is_edge(iiy1, iix1)
                for edge in edges:
                    if edge:
                        A[s, s] = A[s,s] + 1
                for condition in ne_conditions:
                    iiy2 = iiy1 + condition[1]
                    iix2 = iix1 + condition[0]
                    if not self.is_out(iiy2, iix2):
                        s2 = (self.m * iiy2) + iix2
                        A[s, s2] = 1
                        A[s2, s] = 1
        I = scipy.sparse.eye(size)
        # DEBUG A is singular 
        # calculating w3 
        w2_x = w2[:,:,0].reshape(size)
        w2_y = w2[:,:,1].reshape(size)
        w30 = scipy.sparse.linalg.spsolve(I - (self.viscosity * dt * h**2)*A, w2_x)
        w31 = scipy.sparse.linalg.spsolve(I - (self.viscosity * dt * h**2)*A, w2_y)
        w3 = np.zeros([self.n/h, self.m/h, 2])
        w3[:,:,0] = w30.reshape([self.n/h, self.m/h])
        w3[:,:,1] = w31.reshape([self.n/h, self.m/h])

        # OH YEAH ! 
        # i have w3, now I can compute pressure, finally 
        # Qp = div_w3
        div_w3 = self.compute_divergence(w3, h, h)
        div_w3_reshaped = div_w3.reshape(size)
        """
        Q = scipy.sparse.dok_matrix((size, size))
        for ix in ax:
            for iy in ay:
                s = self.m * iy + ix 
                s1 = self.m*(iy + h) + ix 
                s2 = self.m*iy + ix+h
                s3 = self.m*(iy - h) + ix 
                s4 = self.m*iy + ix - h
                # these are not indices yet, divide by h
                s = int(s/h)
                s1 = int(s1/h)
                s2 = int(s2/h)
                s3 = int(s3/h) 
                s4 = int(s4/h)
                #Q[s][s] = -4/(h**2)
                if (ix, iy) in [(0, 0), (0, self.n), (self.m, 0), (self.m, self.n)]:
                    Q[s, s] = 0
                elif ix + h == self.m or self.boundaries[iy, ix + h] or ix - h < 0 or self.boundaries[iy, ix-h]:
                    Q[s, s] = -2/(h**2)
                    Q[s, s1] = 1/h**2
                    Q[s, s3] = 1/h**2
                elif iy + h == self.n or self.boundaries[iy + h, ix] or iy - h < 0 or self.boundaries[iy-h, ix]:
                    Q[s, s2] = 1/h**2
                    Q[s, s4] = 1/h**2
                    Q[s, s] = -2/(h**2)
                else:
                    Q[s, s] = -4/(h**2)
                    Q[s, s1] = 1/h**2 
                    Q[s, s2] = 1/h**2
                    Q[s, s3] = 1/h**2
                    Q[s, s4] = 1/h**2
        """
        p_ = scipy.sparse.linalg.spsolve(A / h**2, div_w3_reshaped)
        p = p_.reshape(self.n/h, self.m/h)
        grad_p = self.compute_gradient(p, h, h)
        w4 = w3 - grad_p 
        self.velocities = w4 

        # copy w1 to old 
        self.old = w1.copy()
        self.pressure = p 

        # This is wrong but i use it because i can and it is easy to debug with existing animators 
        self.densities = p 
