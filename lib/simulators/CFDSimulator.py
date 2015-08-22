
import numpy as np
import numpy.linalg 

from .BaseSimulator import BaseSimulator

import scipy.interpolate as interpolate
import scipy.sparse
import scipy.sparse.linalg

import os.path

class CFDSimulator(BaseSimulator):
    
    def get_boundaries_hash(self):
        if (self.boundaries == 0).all():
            return ""
        else:
            return str(hash(self.boundaries.tostring()))

    def save_sparse_csr(self, filename,array):
        a = array.tocsr()
        np.savez(filename,data = a.data ,indices=a.indices, indptr =a.indptr, shape=a.shape)

    def load_sparse_csr(self, filename):
        loader = np.load(filename)
        return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader["shape"]).tolil()

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

    def get_scaling_condition(self, scale=10):
        n,m = (int(self.n), int(self.m))
        cond = np.zeros([int(self.n*self.m), int(self.n*self.m)], dtype=np.bool)
        for i in range(n):
            for j in range(m):
                for k in range(n):
                    for l in range(m):
                        if i % (scale+1) == 0 and j % (scale+1) == 0 and k % (scale+1) == 0 and l % (scale+1) == 0:
                            cond[int(i*self.m) + j, int(k*self.m) + l] = True
        return cond

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
                if not self.boundary(iiy1, iix1):
                    A[s,s] = -4
                    A[s, s+1] = 1
                    A[s, s-1] = 1
                    A[s,s+c] = 1 
                    A[s,s-c] = 1
        self.save_sparse_csr(cache_filename, A)
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
        A_cache_filename = "cache/Av-%d-%d-%f%s.npz" % (self.n, self.m, self.h, self.get_boundaries_hash())
        bx_cache_filename = "cache/bv_x-%d-%d-%f%s.npz" % (self.n, self.m, self.h, self.get_boundaries_hash())
        by_cache_filename = "cache/bv_y-%d-%d-%f%s.npz" % (self.n, self.m, self.h, self.get_boundaries_hash())
        if os.path.isfile(A_cache_filename) and os.path.isfile(bx_cache_filename) and os.path.isfile(by_cache_filename):
            A = self.load_sparse_csr(A_cache_filename)
            bx = np.array(self.load_sparse_csr(bx_cache_filename).todense()).reshape(self.size)
            by = np.array(self.load_sparse_csr(by_cache_filename).todense()).reshape(self.size)
            return (A,bx,by)
        A = M.copy()
        b_x = np.ones(self.size)
        b_y = np.ones(self.size)
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

    def reset_solid_gradient(self, gp_):
        gp = gp_.copy()
        n,m,d = gp.shape 
        for i in range(n):
            for j in range(m):
                if self.boundaries[i,j]:
                    gp[i,j,0] = 0 
                    gp[i,j,1] = 0
                if self.boundary_left(i,j) or self.boundary_right(i,j):
                    gp[i,j,0] = 0
                if self.boundary_up(i,j) or self.boundary_down(i,j):
                    gp[i,j,1] = 0
        return gp 

    def add_force(self, v, dt, f):
        return v + dt*f

    def scale_down(self, A, b, scale=10):
        cond = None
        if scale != 10:
            cond = self.get_scaling_condition(scale=scale)
        else:
            cond = self.cond
        row_step = int(self.n/scale) + 1
        column_step = int(self.m / scale) + 1
        step = int(self.size/scale**2) + 1
        b_normal = b.reshape(int(self.n), int(self.m))
        mark_x = np.zeros([int(self.n), int(self.m)])
        mark_x[:,::column_step].fill(1)
        mark_y = np.zeros([int(self.n), int(self.m)])
        mark_y[::row_step, :].fill(1)
        condition = np.logical_and(mark_x,mark_y)

        c = b_normal[condition]

        #A4D = A.reshape([self.n, self.m, self.n, self.m])
        B = None 
        if scale == 10:
            B = self.Ap_scaled
        else:
            B = A[cond].todense().reshape([scale**2, scale**2])
        #B = A[0:self.n:row_step, 0:self.m:column_step, 0:self.n:row_step, 0:self.m:column_step].reshape([100, 100])
        return (B, c)

    def scale_up(self, p, scale=10):
        p = p.reshape(scale, scale)
        ax = np.linspace(0, self.m-1, scale)
        ay = np.linspace(0, self.n-1, scale)
        func = scipy.interpolate.RectBivariateSpline(ax,ay, p)
        
        y,x = np.mgrid[0:int(self.n), 0:int(self.m)]

        return func.ev(y,x)
    
    def scale_down_field(self, b, scale=10):
        row_step = int(self.n/scale) + 1
        column_step = int(self.m / scale) + 1
        b_normal = b.reshape(int(self.n), int(self.m))
        mark_x = np.zeros([int(self.n), int(self.m)]).astype(bool)
        mark_x[:,::column_step] = True
        mark_y = np.zeros([int(self.n), int(self.m)]).astype(bool)
        mark_y[::row_step, :] = True
        condition = np.logical_and(mark_x,mark_y)

        c = b_normal[condition].reshape([scale, scale])
        return c

    def scale_up_field(self, p, scale=10):
        p = p.reshape(scale, scale)
        ax = np.linspace(0, self.m-1, scale)
        ay = np.linspace(0, self.n-1, scale)
        func = scipy.interpolate.RectBivariateSpline(ax,ay, p)
        return func.ev(self.y, self.x)
    
    def scale_boundaries(self, scale=10):
        self.tmp_boundaries = self.boundaries.copy()
        #self.boundaries = self.scale_down_field(self.boundaries, scale=scale)
        b = np.zeros([scale, scale], dtype=np.bool)
        stepr = int(self.n / scale) + 1
        stepc = int(self.m / scale) + 1
        for i in range(0, scale):
            for j in range(0, scale):
                if self.boundaries[i*stepr, j*stepc]:
                    b[i,j] = True
        for i in range(1, scale):
            for j in range(1, scale):
                has_solids = self.boundaries[(i-1)*stepr+1:i*stepr, (j-1)*stepc+1:j*stepc].any()
                if has_solids:
                    b[i,j] = True
                    b[i-1,j] = True
                    b[i,j-1] = True
                    b[i-1,j-1] = True
        self.boundaries = b 
        self.logger.print_vector("Scaled boundaries", b)

    def rescale_boundaries(self):
        self.boundaries = self.tmp_boundaries.copy()

    def poisson(self, A, w):
        """
        #S = A * scipy.sparse.identity(A.shape[0])
        S = np.tril(A)
        # T = S - A

        # S = np.tril(A)
        T = S - A
        # B = scipy.sparse.linalg.inv(S).dot(T)
        # el, ev = scipy.sparse.linalg.eigs(B)
        N = A.shape[0]
        p = np.zeros(N)
        w = np.array(w).reshape(N)
        for i in range(40):
            Tp = np.array(T.dot(p)).reshape(N)
            p = np.linalg.solve(S, w + Tp)
        """
        p = np.linalg.solve(A, w)
        return p

    def projection(self, w3, dt):
        scale = 10
        div_w3 = self.compute_divergence(w3, self.h, self.h, edge_order=1)
        div_w3_reshaped = div_w3.reshape(self.size)
        div_w3_scaled = self.scale_down_field(div_w3_reshaped)
        self.scale_boundaries(scale=scale)
        M, c = self.pressure_boundaries(div_w3_scaled.reshape(scale**2))
        p = self.poisson(M, c)
        #p = p_.reshape(int(self.n/self.h), int(self.m/self.h))
        # Set boundaries back 
        p = p.reshape(scale,scale)
        
        p = self.scale_up_field(p, scale=scale)
        self.rescale_boundaries()
        grad_p = self.compute_gradient(p, self.h, self.h, edge_order=1)
        self.logger.print_vector("p = ", p)
        self.logger.print_vector("grad p_x = ", grad_p[:,:,0])
        self.logger.print_vector("grad p_y = ", grad_p[:,:,1])
        w4 = w3 - grad_p 
        return (w4, p)

    def advect_substance(self, u, h, path, dt):
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
        res = h - s*dt*a + k*lh*dt
        self.rescale_boundaries()
        return self.scale_up_field(res) * self.non_boundaries

    def start(self):
        self.deltas = []

        self.h = 1.0
        self.n, self.m = (self.velocities.shape[0]*self.h, self.velocities.shape[1]*self.h)
        
        self.non_boundaries = np.ones([int(self.n), int(self.m)]) - self.boundaries
        
        self.forces = 10*self.velocities 
        self.velocities.fill(0)
        # Set forces :)))
        #for fi in range(int(self.n/self.h)):
        #    self.forces[fi, :, 0] = np.linspace(100, 0, self.m/self.h)
        
        self.y, self.x = np.mgrid[0:int(self.n), 0:int(self.m)]
        self.A, self.I, self.size = self.get_laplacian_operator()
        self.bmap = np.zeros([int(self.n/self.h), int(self.m/self.h)])
        self.iteration = 0
        
        self.Av, self.bv_x, self.bv_y = self.get_velocity_laplacian_operator(self.A)
    
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

        #self.densities = self.advect_substance(self.velocities, self.densities, self.path, dt)
        
        self.logger.print_vector("div v", self.compute_divergence(self.velocities, self.h, self.h))
        
        self.logger.print_vector("divergence error", np.abs(self.compute_divergence(self.velocities, self.h, self.h)).max())

        #self.forces.fill(0)
        self.logger.print_vector("Substance sum: ", self.densities.sum())
        self.logger.print_vector("Substance: ", self.densities)
        self.iteration += 1
