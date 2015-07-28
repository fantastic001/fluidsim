
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
    DEBUG_INTERACTIVE_PLOTS_FIELD = False
    DEBUG_INTERACTIVE_PLOTS_SPEED = True
    
    def get_boundaries_hash(self):
        if (self.boundaries == 0).all():
            return ""
        else:
            return str(hash(self.boundaries.tostring()))

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
        if self.DEBUG_INTERACTIVE_PLOTS_FIELD:
            plt.quiver(v[::10,::10, 0], v[::10,::10,1], units="width")
            plt.title(title)
            plt.show()
            plt.clf()
        if self.DEBUG_INTERACTIVE_PLOTS_SPEED:
            plt.title(title)
            speed = np.sqrt(v[:,:,0]**2 + v[:,:,1]**2)
            plt.imshow(speed)
            plt.show()

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
        cache_filename = "cache/A-%d-%d-%f%s.npz" % (self.n, self.m, self.h, self.get_boundaries_hash())
        
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
        A_cache_filename = "cache/Ap-%d-%d-%f%s.npz" % (self.n, self.m, self.h, self.get_boundaries_hash())
        b_cache_filename = "cache/bp-%d-%d-%f%s.npz" % (self.n, self.m, self.h, self.get_boundaries_hash())
        if os.path.isfile(A_cache_filename) and os.path.isfile(b_cache_filename):
            A = self.load_sparse_csr(A_cache_filename)
            b = self.load_sparse_csr(b_cache_filename)
            return (A,b)
        A = M.tolil()
        b = np.ones(self.size)
        columns = int(self.m/self.h)
        rows = int(self.n/self.h)
        for i in range(rows):
            for j in range(columns):
                s = columns * i + j
                self.print_vector("Pressure laplacian index: ", s)
                if self.boundary(i,j):
                    A[s,s] = -2
                    if self.boundary_edge(i,j) or self.boundaries[i,j]:
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
        A = A.tocsr()
        self.save_sparse_csr(A_cache_filename, A)
        self.save_sparse_csr(b_cache_filename, A)
        return (A, b)

    def pressure_boundaries(self, c):
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
                if self.boundaries[i,j]:
                    A[s,s] = 1
                    b_x[s] = 0
                    b_y[s] = 0
                if self.boundary(i,j):
                    A[s,s] = 1
                    #b_x[s] = 0
                    #b_y[s] = 0
                    if self.boundary_up(i,j) or self.boundary_down(i,j):
                        b_y[s] = 0
                    if self.boundary_right(i,j) or self.boundary_left(i,j):
                        b_x[s] = 0
        return (A,b_x, b_y)
    
    def reset_solid_velocities(self, v):
        w = v.copy()
        n,m,d = v.shape
        for i in range(n):
            for j in range(m):
                if self.boundaries[i,j]:
                    w[i,j] = 0
        return w 
    
    def reset_edge_pressure(self, p_):
        p = p_.copy()
        n,m = p.shape
        for i in range(n):
            for j in range(m):
                if self.boundary_left(i,j):
                    p[i,j] = p_[i,j+1]
                elif self.boundary_right(i,j):
                    p[i,j] = p_[i,j-1]
                elif self.boundary_up(i,j):
                    p[i,j] = p_[i+1,j]
                elif self.boundary_down(i,j):
                    p[i,j] = p_[i-1,j]
                else:
                    pass
        return p 

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
        s[:,:,0] = u_x * duxdx + u_y * duxdy
        s[:,:,1] = u_x * duydx + u_y * duydy
        return s

    def add_force(self, v, dt, f):
        return v + dt*f

    def advection(self, w1, dt, path):
        w2 = w1.copy()
        for j in range(int(self.m)):
            for i in range(int(self.n)):
                psi = path[i,j,1]
                psj = path[i,j,0]
                if psi < 0 or psi >= self.n or psj < 0 or psj >= self.m:
                    #w2[i,j,0] = 0
                    #w2[i,j,1] = 0
                    pass
                else:
                    w2[psi,psj,0] = w1[i, j, 0]
                    w2[psi,psj,1] = w1[i, j, 1]
        return w2
        
    def update_path(self, path, w1, dt):
        # set new paths 
        newpath = path.copy()
        for i in range(int(self.n)):
            for j in range(int(self.m)):
                if not (path[i,j,0] < 0 or path[i,j,0] >= self.m or path[i,j,1] < 0 or path[i,j,1] >= self.n):
                    newpath[i,j,0] += dt*w1[path[i,j,1], path[i,j,0], 0]
                    newpath[i,j,1] += dt*w1[path[i,j,1], path[i,j,0], 1]
        return newpath

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
    
    def scale_down_field(self, b, scale=10):
        row_step = int(self.n/scale)
        column_step = int(self.m / scale)
        step = int(self.size/scale**2)
        b_normal = b.reshape(self.n, self.m)
        c = b_normal[0:self.n:row_step, 0:self.m:column_step]
        return c

    def scale_up_field(self, p, scale=10):
        p = p.reshape(scale, scale)
        x = np.linspace(0, self.m, scale)
        y = np.linspace(0, self.n, scale)
        func = scipy.interpolate.RectBivariateSpline(x,y, p)
        return func(np.linspace(0, self.m, self.m), np.linspace(0, self.n, self.n))
    
    def scale_boundaries(self, scale=10):
        self.tmp_boundaries = self.boundaries.copy()
        self.boundaries = self.scale_down_field(self.boundaries, scale=scale)

    def rescale_boundaries(self):
        self.boundaries = self.tmp_boundaries.copy()

    def diffusion(self, w2, dt):
        scale = 10
        w2_x = w2[:,:,0].reshape(self.size)
        w2_y = w2[:,:,1].reshape(self.size)
        M, c_x, c_y = self.velocity_boundaries(self.A, w2_x, w2_y)
        L = self.I - (self.viscosity * dt)*M
        self.scale_boundaries(scale=scale)
        L1, c_x = self.scale_down(L, c_x)
        L2, c_y = self.scale_down(L, c_y)
        self.print_vector("L1", L1.todense())
        self.print_vector("L2", L2.todense())
        self.print_vector("c_x", c_x)
        self.print_vector("c_y", c_y)
        w30 = scipy.sparse.linalg.spsolve(L1, c_x)
        w31 = scipy.sparse.linalg.spsolve(L2, c_y)
        
        w30 = self.scale_up(w30)
        w31 = self.scale_up(w31)
        self.rescale_boundaries()

        w3 = np.zeros([int(self.n/self.h), int(self.m/self.h), 2])
        w3[:,:,0] = w30.reshape([int(self.n/self.h), int(self.m/self.h)])
        w3[:,:,1] = w31.reshape([int(self.n/self.h), int(self.m/self.h)])
        
        # at boundaries, velocity is zero 
        #w3[0,:, 0] = 0
        #w3[-1,:, 0] = 0
        #w3[:, 0, 0] = 0
        #w3[:,-1, 0] = 0

        #w3[0,:, 1] = 0
        #w3[-1,:, 1] = 0
        #w3[:,0, 1] = 0
        #w3[:,-1, 1] = 0
        
        return w3

    def poisson(self, A, w):
        #S = A * scipy.sparse.identity(A.shape[0])
        S = scipy.sparse.tril(A)
        # T = S - A

        # S = np.tril(A)
        T = S - A
        self.print_vector("S: ", S.todense())
        # B = scipy.sparse.linalg.inv(S).dot(T)
        # el, ev = scipy.sparse.linalg.eigs(B)
        # self.print_vector("Eigenvalues of inv(S)T: ", el)
        # self.print_vector("Rate of convergence: ", el.max())
        # self.print_vector("W: ", w)
        p = np.zeros(A.shape[0])
        for i in range(10):
            p = scipy.sparse.linalg.spsolve(S, w + T.dot(p))
            # p = np.linalg.solve(S, w + T.dot(p))
            # self.print_vector("Pressure: ", p)
            # test = w.reshape([self.n, self.m]) - self.compute_divergence(self.compute_gradient(p.reshape([self.n, self.m]), self.h, self.h), self.h, self.h)
            # self.print_vector("divergence: ", test) 
        return p

    def projection(self, w3, dt):
        scale = 10
        self.print_vector("w3: ", w3)
        div_w3 = self.compute_divergence(w3, self.h, self.h, edge_order=1)
        div_w3_reshaped = div_w3.reshape(self.size)
        M, c = self.pressure_boundaries(div_w3_reshaped)
        self.print_vector("Pressure laplacian before scaling: ", M.todense())
        #if (M.todense() == M.todense().transpose()).all():
        #    print("M is symmetric")
        #diag_M = M * scipy.sparse.identity(M.shape[0])
        #P = scipy.sparse.linalg.inv(diag_M).dot(M)
        
        #p_ = scipy.sparse.linalg.spsolve(M, c)
        #np.savetxt("A.csv", M.todense(), delimiter=",")
        #np.savetxt("b.csv", c, delimiter=",")
        #exit(0)
        self.scale_boundaries(scale=scale)
        M_, c_ = self.scale_down(M, c, scale=scale)
        self.print_vector("Scaled laplacian operator: ", M_)
        self.print_vector("Scaled div(w3) operator: ", c_)
        p_ = self.poisson(dt*M_, c_)
        """
        for s in range(self.size):
            p_[s] = p_[s] / diag_M[s,s]
        """
        #p = p_.reshape(int(self.n/self.h), int(self.m/self.h))
        # Set boundaries back 
        p_ = p_.reshape(10,10)
        p_ = self.reset_edge_pressure(p_)

        p = self.scale_up(p_, scale=scale)
        self.rescale_boundaries()
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

    def advect_substance(self, u, h, path, dt):
        """
        h1 = h.copy()
        for j in range(int(self.m)):
            for i in range(int(self.n)):
                psi = path[i,j,1]
                psj = path[i,j,0]
                if psi < 0 or psi >= self.n or psj < 0 or psj >= self.m:
                    #w2[i,j,0] = 0
                    #w2[i,j,1] = 0
                    pass
                else:
                    h1[psi,psj] = h[i, j]
        return h1
        """
        # diffusion constant 
        s = 1
        k = 1
        self.scale_boundaries()
        ux = self.scale_down_field(u[:,:,0])
        uy = self.scale_down_field(u[:,:,1])
        h = self.scale_down_field(h)
        grad_h = self.compute_gradient(h, self.h, self.h)
        a = ux*grad_h[:,:,0] + uy*grad_h[:,:,1]
        uxh = h*ux
        uyh = h*uy
        uh = np.zeros([h.shape[0], h.shape[1], 2])
        uh[:,:,0] = uxh
        uh[:,:,1] = uyh
        duh = self.compute_divergence(uh, self.h, self.h)
        lh = self.compute_divergence(self.compute_gradient(h, self.h, self.h), self.h, self.h)
        self.print_vector("Substance laplacian: ", lh)
        self.print_vector("Substance advection: ", duh)
        res = h - s*dt*a + k*lh*dt
        self.print_vector("Resulting substance field: ", res)
        self.rescale_boundaries()
        return self.scale_up_field(res) * self.non_boundaries

    def start(self):
        self.deltas = []

        self.h = 1.0
        self.n, self.m = (self.velocities.shape[0]*self.h, self.velocities.shape[1]*self.h)

        self.non_boundaries = np.ones([int(self.n), int(self.m)]) - self.boundaries

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
        self.Ap, self.bp = self.get_pressure_laplacian_operator(self.A)
    
    def finish(self):
        #plt.plot(np.diff(self.deltas))
        #plt.show()
        pass
    
    def step(self, dt):
        w0 = self.velocities 
        self.print_vector("w0 x", w0[:,:,0])
        self.print_vector("w0 y", w0[:,:,1])
        self.print_vector("div w0", self.compute_divergence(w0, self.h, self.h))
        self.bmap, self.iteration = self.plot_change(self.iteration, w0, self.bmap)
        self.plot_field(w0, "w0")

        w1 = self.add_force(w0, dt, self.forces)
        self.print_vector("w1 x", w1[:,:,0])
        self.print_vector("w1 y", w1[:,:,1])
        self.print_vector("div w1", self.compute_divergence(w1, self.h, self.h))
        self.bmap, self.iteration = self.plot_change(self.iteration, w1, self.bmap)
        self.plot_field(w1, "w1")

        #self.path = self.update_path(self.path, w1, dt)
        #w2 = self.advection(w1, dt, self.path)
        w2 = w1.copy()
        w2 =  w1 - dt * self.advection_primitive(w1)
        self.print_vector("w2 x", w2[:,:,0])
        self.print_vector("w2 y", w2[:,:,1])
        self.print_vector("div w2", self.compute_divergence(w2, self.h, self.h))
        self.bmap, self.iteration = self.plot_change(self.iteration, w2, self.bmap)
        self.plot_field(w2, "w2")

        w3 = self.diffusion(w2, dt)
        # w3 = w2 + dt * self.viscosity * self.compute_laplacian(w2, self.h, self.h)
        self.print_vector("w3 x", w3[:,:,0])
        self.print_vector("w3 y", w3[:,:,1])
        self.print_vector("div w3", self.compute_divergence(w3, self.h, self.h))
        self.bmap, self.iteration = self.plot_change(self.iteration, w3, self.bmap)
        self.plot_field(w3, "w3")

        w4, p = self.projection(w3, dt)
        self.print_vector("w4 x", w4[:,:,0])
        self.print_vector("w4 y", w4[:,:,1])
        self.print_vector("div w4", self.compute_divergence(w4, self.h, self.h))
        self.velocities = self.reset_solid_velocities(w4)
        self.pressure = p 
        self.plot_field(w4, "w4")

        self.densities = self.advect_substance(self.velocities, self.densities, self.path, dt)
        
        self.print_vector("div v", self.compute_divergence(self.velocities, self.h, self.h))
        self.bmap, self.iteration = self.plot_change(self.iteration, w4, self.bmap)
        
        self.print_vector("divergence error", np.abs(self.compute_divergence(self.velocities, self.h, self.h)).max())

        self.forces.fill(0)
        self.print_vector("Substance sum: ", self.densities.sum())
        self.print_vector("Substance: ", self.densities)
