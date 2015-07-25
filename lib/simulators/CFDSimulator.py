
import numpy as np
import numpy.linalg 

from .BaseSimulator import BaseSimulator

import scipy.interpolate as interpolate
import scipy.sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt 

import os.path

class CFDSimulator(BaseSimulator):
    
    DEBUG = True
    DEBUG_BREAK = False
    DEBUG_PLOT = False
    DEBUG_INTERACTIVE_PLOTS = False
    
    def save_sparse_csr(self, filename,array):
        np.savez(filename,data = array.data ,indices=array.indices, indptr =array.indptr, shape=array.shape)

    def load_sparse_csr(self, filename):
        loader = np.load(filename)
        return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader["shape"])

    def get_condition_number(self, M):
        return scipy.sparse.linalg.onenormest(M) * scipy.sparse.linalg.onenormest(scipy.sparse.linalg.inv(M))

    def plot_change(self, iteration, v, bmap):
        if self.DEBUG_PLOT:
            plt.imsave("debug-%d.png" % iteration, self.compute_speed(v) - bmap, vmax=2, vmin=-2)
            iteration += 1
            bmap = self.compute_speed(v)
        return (bmap, iteration)

    def plot_field(self, v, title=""):
        if self.DEBUG_INTERACTIVE_PLOTS:
            plt.quiver(v[:,:, 0], v[:,:,1], units="width")
            plt.title(title)
            plt.show()
            plt.clf()
    
    def print_vector(self, s, v, full=False):
        if self.DEBUG:
            print(s)
            if full:
                print(v.tolist())
            else:
                print(v)
            if self.DEBUG_BREAK:
                input()

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
        return i==int(self.n/self.h)-1 or self.boundaries[i+1, j]
    def boundary_left(self, i,j):
        return j==0 or self.boundaries[i, j-1]
    def boundary_right(self, i,j):
        return j==int(self.m/self.h)-1 or self.boundaries[i, j+1]

    def boundary(self, i,j):
        return self.boundary_up(i,j) or self.boundary_left(i,j) or self.boundary_right(i,j) or self.boundary_down(i,j)

    def get_laplacian_operator(self):
        cache_filename = "cache/A-%d-%d-%f.npz" % (self.n, self.m, self.h)
        
        # Computing laplacian operator 
        size = int(self.n/self.h) * int(self.m / self.h)
        c = int(self.m/self.h)
        I = scipy.sparse.eye(size)

        # see if there's cached one 
        if os.path.isfile(cache_filename):
            A = self.load_sparse_csr(cache_filename)
            return (A, I, size)
        
        A = scipy.sparse.lil_matrix((size, size))
        for iiy1 in range(int(self.n)):
            for iix1 in range(int(self.m)):
                s = (self.m * iiy1) + iix1
                self.print_vector("Index: ", s)
                #print("ix iy = %d %d" % (ix1, iy1))
                #print("iix1, iiy1 = %d %d" % (iix1, iiy1))
                #print("index %d" % s)
                if not self.boundary(iiy1, iix1):
                    A[s,s] = -4
                    A[s, s+1] = 1
                    A[s, s-1] = 1
                    A[s,s+c] = 1 
                    A[s,s-c] = 1
        A = A.tocsr()
        self.save_sparse_csr(cache_filename, A)
        return (A, I, size)
    
    def get_pressure_laplacian_operator(self, M):
        A_cache_filename = "cache/Ap-%d-%d-%f.npz" % (self.n, self.m, self.h)
        b_cache_filename = "cache/bp-%d-%d-%f.npz" % (self.n, self.m, self.h)
        if os.path.isfile(A_cache_filename) and os.path.isfile(b_cache_filename):
            A = self.load_sparse_csr(A_cache_filename)
            b = self.load_sparse_csr(b_cache_filename)
            return (A,b)
        A = M.copy()
        b = np.ones(self.size)
        columns = int(self.m/self.h)
        rows = int(self.n/self.h)
        for i in range(rows):
            for j in range(columns):
                s = columns * i + j
                if self.boundary(i,j):
                    A[s,s] = -2
                    if (i,j) in [(0,0), (0, columns-1), (rows-1, 0), (rows-1, columns-1)]:
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
        self.save_sparse_csr(A_cache_filename, A)
        self.save_sparse_csr(b_cache_filename, A)
        return (A, b)

    def pressure_boundaries(self, M, c):
        b = c*self.bp
        return (self.Ap,b)
    
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

    def advection(self, w1, dt, path):
        w2 = w1.copy()
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
    
    def scale_down(self, A, b, scale=10):
        row_step = int(self.n/scale)
        column_step = int(self.m / scale)
        step = int(self.size/scale**2)
        b_normal = b.reshape(self.n, self.m)
        c = b_normal[0:self.n:row_step, 0:self.m:column_step].reshape(scale**2)
        #A4D = A.reshape([self.n, self.m, self.n, self.m])
        B = A[0:self.size:step, 0:self.size:step]
        #B = A[0:self.n:row_step, 0:self.m:column_step, 0:self.n:row_step, 0:self.m:column_step].reshape([100, 100])
        return (B, c)

    def scale_up(self, p, scale=10):
        p = p.reshape(scale, scale)
        x = np.linspace(0, self.m, scale)
        y = np.linspace(0, self.n, scale)
        func = scipy.interpolate.RectBivariateSpline(x,y, p)
        return func(np.linspace(0, self.m, self.m), np.linspace(0, self.n, self.n))

    def poisson(self, A, w):
        #S = A * scipy.sparse.identity(A.shape[0])
        S = scipy.sparse.tril(A)
        # T = S - A

        # S = np.tril(A)
        T = S - A

        # B = scipy.sparse.linalg.inv(S).dot(T)
        # el, ev = scipy.sparse.linalg.eigs(B)
        # self.print_vector("Eigenvalues of inv(S)T: ", el)
        # self.print_vector("Rate of convergence: ", el.max())
        # self.print_vector("W: ", w)
        p = np.zeros(A.shape[0])
        for i in range(20):
            self.print_vector("shapes, ", [p.shape, w.shape])
            p = scipy.sparse.linalg.spsolve(S, w + T.dot(p))
            # p = np.linalg.solve(S, w + T.dot(p))
            # self.print_vector("Pressure: ", p)
            # test = w.reshape([self.n, self.m]) - self.compute_divergence(self.compute_gradient(p.reshape([self.n, self.m]), self.h, self.h), self.h, self.h)
            # self.print_vector("divergence: ", test) 
        return p

    def projection(self, w3, dt):
        scale = 4
        self.print_vector("w3: ", w3)
        div_w3 = self.compute_divergence(w3, self.h, self.h, edge_order=1)
        div_w3_reshaped = div_w3.reshape(self.size)
        self.print_vector("div w3: ", div_w3_reshaped)
        M, c = self.pressure_boundaries(self.A, div_w3_reshaped)
        #if (M.todense() == M.todense().transpose()).all():
        #    print("M is symmetric")
        #diag_M = M * scipy.sparse.identity(M.shape[0])
        #P = scipy.sparse.linalg.inv(diag_M).dot(M)
        
        #p_ = scipy.sparse.linalg.spsolve(M, c)
        #np.savetxt("A.csv", M.todense(), delimiter=",")
        #np.savetxt("b.csv", c, delimiter=",")
        #exit(0)
        M_, c_ = self.scale_down(M, c, scale=scale)
        self.print_vector("Scaled laplacian operator: ", M_)
        self.print_vector("Scaled div(w3) operator: ", c_)
        p_ = self.poisson(dt*M_, c_)
        """
        for s in range(self.size):
            p_[s] = p_[s] / diag_M[s,s]
        """
        #p = p_.reshape(int(self.n/self.h), int(self.m/self.h))
        p = self.scale_up(p_, scale=scale)
        # Set boundaries back 
        #p[0,:] = p[1,:]
        #p[-1,:] = p[-2, :]
        #p[:,0] = p[:,1]
        #p[:,-1] = p[:,-2]

        grad_p = self.compute_gradient(p, self.h, self.h, edge_order=1)
        #self.print_vector("b = ", c, full=True)
        #self.print_vector("A = ", M.todense(), full=True)
        #self.print_vector("A inverse", scipy.sparse.linalg.inv(M).todense(), full=True)
        self.print_vector("p = ", p)
        self.print_vector("grad p_x = ", grad_p[:,:,0])
        #self.print_vector("condition number of system: ", self.get_condition_number(M))
        #self.print_vector("condition number of preconditioned system: ", self.get_condition_number(P))
        #self.print_vector("Condition number of diagonalized matrix: ", self.get_condition_number(diag_M))
        w4 = w3 - dt*grad_p 
        return (w4, p)

    def start(self):
        self.deltas = []

        self.h = 1.0
        self.n, self.m = (self.velocities.shape[0]*self.h, self.velocities.shape[1]*self.h)
        self.forces = np.zeros([int(self.n/self.h), int(self.m/self.h), 2]) # Will be removed and modeled differently
        # Set forces :)))
        for fi in range(int(self.n/self.h)):
            self.forces[fi, :, 0] = np.linspace(1000, 0, self.m/self.h)
        
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
        self.Ap, self.bp = self.get_pressure_laplacian_operator(self.A)
    
    def finish(self):
        #plt.plot(np.diff(self.deltas))
        #plt.show()
        pass
    
    def step(self, dt):
        w0 = self.velocities 
        self.print_vector("w0", w0[:,:,0])
        self.bmap, self.iteration = self.plot_change(self.iteration, w0, self.bmap)
        self.plot_field(w0, "w0")

        w1 = self.add_force(w0, dt, self.forces)
        self.print_vector("w1", w1[:,:,0])
        self.bmap, self.iteration = self.plot_change(self.iteration, w1, self.bmap)
        self.plot_field(w1, "w1")

        # w2 = self.advection(w1, dt, self.path)
        # self.path = self.update_path(self.path, w1, dt)
        w2 = w1.copy()
        w2 =  w1 - dt * self.advection_primitive(w1)
        self.print_vector("w2", w2[:,:,0])
        self.bmap, self.iteration = self.plot_change(self.iteration, w2, self.bmap)
        self.plot_field(w2, "w2")

        # w3 = self.diffusion(w2, dt)
        w3 = w2 + dt * self.viscosity * self.compute_laplacian(w2, self.h, self.h)
        self.print_vector("w3", w3[:,:,0])
        self.bmap, self.iteration = self.plot_change(self.iteration, w3, self.bmap)
        self.plot_field(w3, "w3")

        w4, p = self.projection(w3, dt)
        self.print_vector("w4", w4[:,:,0])
        self.velocities = w4 
        self.pressure = p 
        self.plot_field(w4, "w4")

        self.densities = p 
        
        self.print_vector("div v", self.compute_divergence(self.velocities, self.h, self.h))
        self.bmap, self.iteration = self.plot_change(self.iteration, w4, self.bmap)
        
        self.forces.fill(0)
