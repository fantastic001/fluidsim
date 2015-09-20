
import unittest 

import numpy as np
import numpy.testing as nptest
import scipy.sparse.linalg 

from ..utils import staggered

class TestStaggered(unittest.TestCase):
    
    def setUp(self):
        n = 100
        self.n = n
        self.normal_constant_u = np.ones([n,n])*100
        self.normal_constant_v = np.ones([n,n])*100

        self.staggered_constant_u = np.ones([n-1, n])*100
        self.staggered_constant_v = np.ones([n, n-1])*100

        self.staggered_zeros_u = np.zeros([99,100])
        self.staggered_zeros_v = np.zeros([100, 99]) 

        x,y = np.mgrid[0:100,0:100]
        x = x[:-1, :]
        y = y[:, :-1]

        self.staggered_x_u = x 
        self.staggered_y_v = y
        
        self.normal_zeros_u = np.zeros([100,100])
        self.normal_zeros_v = np.zeros([100, 100]) 

        x,y = np.mgrid[0:100,0:100]

        self.normal_x_u = x 
        self.normal_y_v = y

        self.psolver = staggered.set_solids(np.zeros([100,100]))

    def test_field_transpose(self):
        u,v = staggered.field_transpose(self.normal_constant_u, self.normal_constant_v)
        nptest.assert_allclose(u, self.normal_constant_u)
        nptest.assert_allclose(v, self.normal_constant_v)

    def test_to_centered(self):
        #ubc, vbc = staggered.attach_boundaries(self.staggered_constant_u, self.staggered_constant_v)
        ubc, vbc = self.staggered_constant_u, self.staggered_constant_v

        u,v = staggered.to_centered(ubc, vbc)
        nptest.assert_allclose(u[1:-1,1:-1], self.normal_constant_u[1:-1,1:-1])
        nptest.assert_allclose(v[1:-1,1:-1], self.normal_constant_v[1:-1,1:-1])
    
    def test_to_staggered(self):
        u,v = staggered.to_staggered(self.normal_constant_u, self.normal_constant_v)
        nptest.assert_allclose(u[1:-1,1:-1], self.staggered_constant_u[1:-1,1:-1])
        nptest.assert_allclose(v[1:-1,1:-1], self.staggered_constant_v[1:-1,1:-1])

        # test real example of symmetric system 
        size = 100
        
        dt = 0.1
        k = 1e-06
        y,x = np.mgrid[0:size, 0:size]
        r = 5
        w = np.zeros([size, size, 2])
        w[:, 0:0.4*size, 0] = np.linspace(size // 10, 0, 40)
        u,v = w[:,:,0], w[:,:,1]
        u,v = staggered.field_transpose(u,v)
        u,v = staggered.to_staggered(u,v)
        nptest.assert_allclose(
            u[:, 0:size//2],
            u[:, -1:size//2-1:-1],
            err_msg="x component not symmetric"
        )
        nptest.assert_allclose(
            v[0:size//2, :],
            v[-1:size//2-1:-1, :],
            err_msg="y component not symmetric"
        )

    def test_transition(self):
        u,v = staggered.to_staggered(self.normal_constant_u, self.normal_constant_v)
        #ubc, vbc = staggered.attach_boundaries(u, v)
        ubc, vbc = u,v
        u,v = staggered.to_centered(ubc, vbc)

        nptest.assert_allclose(u[1:-1,1:-1], self.normal_constant_u[1:-1,1:-1])
        nptest.assert_allclose(v[1:-1,1:-1], self.normal_constant_v[1:-1,1:-1])


    def test_projection(self):
        u,v = 100 * self.staggered_x_u, self.staggered_zeros_v 
        u,v,p = staggered.projection(u,v,self.psolver, 
            np.zeros([100,100])
        )
        
        ubc, vbc = staggered.attach_boundaries(u,v)
        div = staggered.compute_divergence(ubc, vbc) 
        nptest.assert_allclose(
            div[1:-1, 1:-1],
            np.zeros([98,98]),
            atol=1e-09,
            err_msg="Staggered projection does not produce zero divergence in domain"
        )
        nptest.assert_allclose(
            div,
            np.zeros([100,100]),
            atol=1e-09,
            err_msg="Staggered projection does not produce zero divergence on boundaries included"
        )

        dim = 100
        nptest.assert_allclose(
            p[:, 0:dim//2], 
            p[:, dim-1:(dim // 2)-1:-1],
            rtol=0,
            atol=1e-13,
            err_msg="pressure not symmetric"
        )
        np.savetxt("symmetry.csv", u[:, 0:dim//2] - u[:, dim-1:(dim // 2)-1:-1], delimiter=",")
        nptest.assert_allclose(
            u[:, 0:dim//2], 
            u[:, dim-1:(dim // 2)-1:-1],
            atol=1e-10,
            err_msg="staggered projection does not produce symmetry"
        )
    
    def test_compute_divergence(self):
        u,v = 100 * self.staggered_x_u, self.staggered_zeros_v 
        u,v = staggered.attach_boundaries(u,v)
        div = staggered.compute_divergence(u,v)
        dim = 100
        nptest.assert_allclose(
            div[:, 0:dim//2], 
            div[:, dim-1:(dim // 2)-1:-1],
            err_msg="divergence computation does not produce symmetry"
        )
        nptest.assert_allclose(
            div[1:-1, 1:-1], 
            100*np.ones([98,98]),
            err_msg="divergence computation does not work"
        )
    
    def test_set_solids(self):
        div = self.normal_constant_u
        Lp = staggered.set_solids(np.zeros([100,100]), ret_raw=True)
        res = scipy.sparse.linalg.spsolve(Lp, div.reshape(10000))
        dim = 100
        res = res.reshape([100,100])
        nptest.assert_allclose(
            res[:, 0:dim//2], 
            res[:, dim-1:(dim // 2)-1:-1],
            err_msg="Lp computation does not produce symmetry"
        )

        res = self.psolver.solve(div.reshape(10000))
        res = res.reshape([100,100])
        nptest.assert_allclose(
            res[:, 0:dim//2], 
            res[:, dim-1:(dim // 2)-1:-1],
            err_msg="psolver computation does not produce symmetry"
        )

    def test_apply_pressure(self):
        u,v = self.staggered_x_u, self.staggered_zeros_v
        p = self.normal_x_u
        u,v = staggered.apply_pressure(u,v,p)
        nptest.assert_allclose(
            u, 
            self.staggered_x_u + 1,
            err_msg="apply_pressure() does not work"
        )
        
        u,v = self.staggered_x_u, self.staggered_y_v
        p = self.normal_x_u + self.normal_y_v
        u,v = staggered.apply_pressure(u,v,p)
        nptest.assert_allclose(
            u, 
            self.staggered_x_u + 1,
            err_msg="apply_pressure() does not work for x-component"
        )
        nptest.assert_allclose(
            v, 
            self.staggered_y_v + 1,
            err_msg="apply_pressure() does not work for y-component"
        )
