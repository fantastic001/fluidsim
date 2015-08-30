
from ..simulators import CFDSimulator, CFDImplicitSimulator, CFDExplicitSimulator
from ..structures import Cell 

import unittest 

import numpy as np
import numpy.testing as nptest


class TestCFDSimulator(unittest.TestCase):
    
    def setUp(self):
        self.zeros10 = np.zeros([10,10])
        self.zeros100 = np.zeros([100,100])
        self.zeros102 = np.zeros([10,10,2])
        self.zeros1002 = np.zeros([100,100,2])

    def test_start(self):
        simulator = CFDSimulator(np.zeros([10, 10]), np.zeros([10,10,2]), np.zeros([10,10]), 0.0001)
        self.assertEqual(simulator.A.shape, (100, 100))

    def test_gradient(self):
        v0 = np.zeros([100, 100, 2])
        p0 = np.zeros([100,100])
        simulator = CFDSimulator(p0, v0, p0,0) # we do not need to specify fluid properties for this test
        y,x = np.mgrid[0:100:0.1, 0:100:0.1]
        f = x + 2*y 
        grad = simulator.compute_gradient(f, 0.1, 0.1)
        dfdx = grad[:,:,0]
        dfdy = grad[:,:,1]
        nptest.assert_array_almost_equal(dfdy, 2*np.ones(f.shape))
        nptest.assert_array_almost_equal(dfdx, np.ones(f.shape))
        
        y,x = np.mgrid[0:100:0.1, 0:100:0.1]
        f = x + y**2
        grad = simulator.compute_gradient(f, 0.1, 0.1)
        dfdx = grad[:,:,0]
        dfdy = grad[:,:,1]
        nptest.assert_array_almost_equal(dfdy, 2*y, decimal=1)
        nptest.assert_array_almost_equal(dfdx, np.ones(f.shape))
        
        y,x = np.mgrid[0:100:0.1, 0:100:0.1]
        f = x**2 + y**2
        grad = simulator.compute_gradient(f, 0.1, 0.1)
        dfdx = grad[:,:,0]
        dfdy = grad[:,:,1]
        nptest.assert_array_almost_equal(dfdy, 2*y, decimal=3)
        nptest.assert_array_almost_equal(dfdx, 2*x, decimal=3)

        # test with delta_x and delta_y
        y,x = np.mgrid[0:100:0.1, 0:100:0.1]
        f = x + y**2
        grad = simulator.compute_gradient(f, dx=0.1, dy=0.1, delta_x=0.05, delta_y=0.05)
        dfdx = grad[:,:,0]
        dfdy = grad[:,:,1]
        nptest.assert_array_almost_equal(dfdy, 2*y, decimal=1, verbose=True)
        nptest.assert_array_almost_equal(dfdx, np.ones(f.shape), verbose=True)
        
        y,x = np.mgrid[0:100:0.1, 0:100:0.1]
        f = x**2 + y**2
        grad = simulator.compute_gradient(f, dx=0.1, dy=0.1, delta_x=0.05, delta_y=0.05)
        dfdx = grad[:,:,0]
        dfdy = grad[:,:,1]
        nptest.assert_array_almost_equal(dfdy[1:-1, 1:-1], 2*y[1:-1,1:-1], decimal=3, verbose=True)
        nptest.assert_array_almost_equal(dfdx[1:-1, 1:-1], 2*x[1:-1,1:-1], decimal=3, verbose=True)

    def test_divergence(self):
        y,x = np.mgrid[0:100:0.1, 0:100:0.1]
        expected = np.zeros([x.shape[0], x.shape[1]])
        f = np.zeros([x.shape[0], x.shape[1], 2])
        f[:, :, 0] = x
        f[:,:,1] = y
        simulator = CFDSimulator(np.zeros([100,100]),np.zeros([100,100,2]),np.zeros([100,100]),0)
        divergence = simulator.compute_divergence(f, 0.1, 0.1)
        nptest.assert_array_almost_equal(divergence, 2*np.ones(x.shape))

        f[:,:,0] = y 
        f[:,:,1] = x
        divergence = simulator.compute_divergence(f, 0.1, 0.1)
        nptest.assert_array_almost_equal(divergence, np.zeros(x.shape))
        
        f[:,:,0] = 5*x 
        f[:,:,1] = y
        divergence = simulator.compute_divergence(f, 0.1, 0.1)
        nptest.assert_array_almost_equal(divergence, 6*np.ones(x.shape))

        f[:,:,0] = 2*x**2
        f[:,:,1] = 4*y**2
        expected[:,:] = 4*x + 8*y
        divergence = simulator.compute_divergence(f, 0.1, 0.1)
        nptest.assert_array_almost_equal(divergence, expected)

        # test linearity
        f[:,:,0] = 2*x**2
        f[:,:,1] = 4*y**2
        g = f.copy()
        divergence = simulator.compute_divergence(f+g, 0.1, 0.1)
        divergence_f = simulator.compute_divergence(f, 0.1, 0.1)
        divergence_g = simulator.compute_divergence(g, 0.1, 0.1)
        nptest.assert_array_almost_equal(divergence, divergence_f + divergence_g)
        
        f[:,:,0] = 2*x**2
        f[:,:,1] = 4*y**2
        divergence = simulator.compute_divergence(f, 0.1, 0.1)
        divergence2 = simulator.compute_divergence(2*f, 0.1, 0.1)
        nptest.assert_array_almost_equal(2*divergence, divergence2)

        # test with deltas 

        f[:,:,0] = y 
        f[:,:,1] = x
        divergence = simulator.compute_divergence(f, 0.1, 0.1, delta_x=0.05, delta_y=0.05)
        nptest.assert_array_almost_equal(divergence[1:-1, 1:-1], np.zeros(x.shape)[1:-1, 1:-1])
        
        f[:,:,0] = 5*x 
        f[:,:,1] = y
        divergence = simulator.compute_divergence(f, 0.1, 0.1, delta_x=0.05, delta_y=0.05)
        nptest.assert_array_almost_equal(divergence[1:-1, 1:-1], 6*np.ones(x.shape)[1:-1, 1:-1])

        f[:,:,0] = 2*x**2
        f[:,:,1] = 4*y**2
        expected[:,:] = 4*x + 8*y
        divergence = simulator.compute_divergence(f, 0.1, 0.1, delta_x=0.05, delta_y=0.05)
        nptest.assert_array_almost_equal(divergence[1:-1, 1:-1], expected[1:-1, 1:-1])
    
    def test_laplacian(self):
        y,x = np.mgrid[0:100:0.1, 0:100:0.1]
        f = np.zeros([x.shape[0], x.shape[1], 2])
        f[:,:,0] = x**2
        f[:,:,1] = y**2
        expected = 2*np.ones([f.shape[0], f.shape[1], 2])
        simulator = CFDSimulator(np.zeros([100,100]),np.zeros([100,100,2]),np.zeros([100,100]),0)
        res = simulator.compute_laplacian(f, 0.1, 0.1)
        nptest.assert_array_almost_equal(expected, res, decimal=3)


    def test_laplacian_operator(self):
        simulator = CFDSimulator(np.zeros([10,10]),np.zeros([10,10, 2]),np.zeros([10,10]),1)
        n,m = int(10/simulator.h), int(10/simulator.h)
        # construct grid first 
        y, x = np.mgrid[0:10:simulator.h, 0:10:simulator.h]
        u = np.zeros([int(10/simulator.h), int(10/simulator.h)])
        expected = np.zeros([int(10/simulator.h),int(10/simulator.h)])
        A, I, size = simulator.get_laplacian_operator()

        u[:,:] = 100
        res = A.dot(u.reshape(size))
        expected[:,:] = 0
        nptest.assert_array_almost_equal(res.reshape(n,m), expected)

        u[:,:] = x
        res = A.dot(u.reshape(size))
        expected[:,:] = 0
        nptest.assert_array_almost_equal(res.reshape(n,m)[1:-1,1:-1], expected[1:-1, 1:-1])
        
        u[:,:] = y
        res = A.dot(u.reshape(size))
        expected[:,:] = 0
        nptest.assert_array_almost_equal(res.reshape(n,m)[1:-1,1:-1], expected[1:-1,1:-1])

        u[:,:] = x**2
        res = A.dot(u.reshape(size))
        expected[:,:] = 2
        nptest.assert_array_almost_equal(res.reshape(n,m)[1:-1,1:-1], expected[1:-1,1:-1])
        
        u[:,:] = y**2
        res = A.dot(u.reshape(size))
        expected[:,:] = 2
        nptest.assert_array_almost_equal(res.reshape(n,m)[1:-1,1:-1], expected[1:-1,1:-1])
        
        u[:,:] = x*y
        res = A.dot(u.reshape(size))
        expected[:,:] = 0
        nptest.assert_array_almost_equal(res.reshape(n,m)[1:-1,1:-1], expected[1:-1,1:-1])
        
        u[:,:] = x**2 + y**2
        res = A.dot(u.reshape(size))
        expected[:,:] = 4
        nptest.assert_array_almost_equal(res.reshape(n,m)[1:-1,1:-1], expected[1:-1,1:-1])
        
        u[:,:] = x**2 * y**2
        res = A.dot(u.reshape(size))
        expected[:,:] = 2*x**2 + 2*y**2
        nptest.assert_array_almost_equal(res.reshape(n,m)[1:-1,1:-1], expected[1:-1,1:-1])
    
    def test_pressure_laplacian_operator(self):
        simulator = CFDSimulator(np.zeros([10,10]),np.zeros([10,10, 2]),np.zeros([10,10]),1)
        n,m = int(10/simulator.h), int(10/simulator.h)
        # construct grid first 
        y, x = np.mgrid[0:10:simulator.h, 0:10:simulator.h]
        u = np.zeros([int(10/simulator.h), int(10/simulator.h)])
        expected = np.zeros([int(10/simulator.h),int(10/simulator.h)])
        size = n*m
        A, c = simulator.get_pressure_laplacian_operator()

        u[:,:] = 100
        res = A.dot(u.reshape(size))
        expected[:,:] = 0
        nptest.assert_array_almost_equal(res.reshape(n,m)[1:-1,1:-1], expected[1:-1,1:-1])

        u[:,:] = x
        res = A.dot(u.reshape(size))
        expected[:,:] = 0
        nptest.assert_array_almost_equal(res.reshape(n,m)[1:-1,1:-1], expected[1:-1, 1:-1])
        
        u[:,:] = y
        res = A.dot(u.reshape(size))
        expected[:,:] = 0
        nptest.assert_array_almost_equal(res.reshape(n,m)[1:-1,1:-1], expected[1:-1,1:-1])

        u[:,:] = x**2
        res = A.dot(u.reshape(size))
        expected[:,:] = 2
        nptest.assert_array_almost_equal(res.reshape(n,m)[1:-1,1:-1], expected[1:-1,1:-1])
        
        u[:,:] = y**2
        res = A.dot(u.reshape(size))
        expected[:,:] = 2
        nptest.assert_array_almost_equal(res.reshape(n,m)[1:-1,1:-1], expected[1:-1,1:-1])
        
        u[:,:] = x*y
        res = A.dot(u.reshape(size))
        expected[:,:] = 0
        nptest.assert_array_almost_equal(res.reshape(n,m)[1:-1,1:-1], expected[1:-1,1:-1])
        
        u[:,:] = x**2 + y**2
        res = A.dot(u.reshape(size))
        expected[:,:] = 4
        nptest.assert_array_almost_equal(res.reshape(n,m)[1:-1,1:-1], expected[1:-1,1:-1])
        
        u[:,:] = x**2 * y**2
        res = A.dot(u.reshape(size))
        expected[:,:] = 2*x**2 + 2*y**2
        nptest.assert_array_almost_equal(res.reshape(n,m)[1:-1,1:-1], expected[1:-1,1:-1])
        
    
    def test_projection(self):
        # WARNING: This test assumes correct compute_divergence method
        # TODO Fix this to be independent
        dim = 100
        simulator = CFDSimulator(np.zeros([dim,dim]),np.zeros([dim,dim, 2]),np.zeros([dim,dim]),0.001)
        # construct grid first 
        y, x = np.mgrid[0:dim:simulator.h, 0:dim:simulator.h]
        u = np.zeros([int(dim/simulator.h), int(dim/simulator.h), 2])
        expected = np.zeros([int(dim/simulator.h),int(dim/simulator.h)])
        dt = 0.1

        u[:,:,0] = 1e+15
        u[:,:,1] = 1e+15
        w, p = simulator.projection(u, dt)
        nptest.assert_allclose(simulator.compute_divergence(w, delta_x=0.5, delta_y=0.5), expected, atol=1, rtol=0.1,
            err_msg="Not equal on constant field"
        )
        
        u[:,:,0] = x**2 + y**2
        u[:,:,1] = 2*x**2 + y**2
        w, p = simulator.projection(u, dt)
        grad_p = simulator.compute_gradient(p, delta_x=0.5, delta_y=0.5)
        div_u = simulator.compute_divergence(u, delta_x=0.5, delta_y=0.5)
        div_grad_p = simulator.compute_divergence(grad_p, delta_x=0.5, delta_y=0.5)
        nptest.assert_allclose(div_u - div_grad_p, expected, atol=20, rtol=0.1, 
            err_msg="Not equal when computed divergence on each."
        )
        
        u[:,:,0] = x**2 + y**2
        u[:,:,1] = 2*x**2 + y**2
        w, p = simulator.projection(u, dt)
        nptest.assert_allclose(simulator.compute_divergence(w, delta_x=0.5, delta_y=0.5), expected, atol=1000, rtol=0.1,
            err_msg="Not eqaul on whole for radial field"
        )

        u[:,:,0] = x
        u[:,:,1] = y
        w, p = simulator.projection(u, dt)
        nptest.assert_allclose(simulator.compute_divergence(w, delta_x=0.5, delta_y=0.5), expected, atol=10, rtol=0.1,
            err_msg="Not equal on constant field"
        )
        
        # we expect symmatric response to symmetric input
        u[:,:,0] = x
        u[:,:,1] = 0
        w, p = simulator.projection(u, dt)
        nptest.assert_allclose(w[0:dim // 2, :, 0], w[dim-1:(dim // 2)-1:-1, :, 0], atol=0.01, rtol=0.001,
            err_msg="Not symmetric on field <x, 0>"
        )
        
        u[:,:,0] = 0
        u[:,:,1] = y
        w, p = simulator.projection(u, dt)
        nptest.assert_allclose(w[:, 0:dim // 2, 1], w[:, dim-1:(dim // 2)-1:-1, 1], atol=0.01, rtol=0.001,
            err_msg="Not symmetric on field <0, y>"
        )

        u[:,:,0] = 10 - x/10
        u[[0, -1], :, 0] = 0
        u[:, [0, -1], 0] = 0
        u[:,:,1] = 0
        w, p = simulator.projection(u, dt)
        nptest.assert_allclose(w[0:dim // 2, :, 0], w[dim-1:(dim // 2)-1:-1, :, 0], atol=0.01, rtol=0.001,
            err_msg="Not symmetric on real field"
        )

    def test_scale_boundaries(self):
        
        dim = 100
        scale = 10
        boundaries = np.zeros([dim, dim]).astype(bool)
        p,v = np.zeros([dim, dim]), np.zeros([dim, dim, 2])
        k = 1
        expected = np.zeros([scale, scale]).astype(bool)
        
        b = boundaries.copy()
        b[[55], [55]] = True
        e = expected.copy()
        e[5, 5] = True
        simulator = CFDSimulator(p,v,b,k)
        simulator.scale_boundaries(scale=scale)
        nptest.assert_array_almost_equal(simulator.boundaries, e)
        
        b = boundaries.copy()
        b[[44, 55], [44, 55]] = True
        e = expected.copy()
        e[[4,5], [4,5]] = True
        simulator = CFDSimulator(p,v,b,k)
        simulator.scale_boundaries(scale=scale)
        nptest.assert_array_almost_equal(simulator.boundaries, e)
        
        b = boundaries.copy()
        b[[44, 55, 98], [44, 55, 98]] = True
        e = expected.copy()
        e[[4,5, 8,8,9,9], [4,5,8,9,8,9]] = True
        simulator = CFDSimulator(p,v,b,k)
        simulator.scale_boundaries(scale=scale)
        nptest.assert_array_almost_equal(simulator.boundaries, e)
        
        y,x = np.mgrid[0:dim, 0:dim]
        r = 5
        b = (x - 0.495*dim)**2 + (y - 0.495*dim)**2 <= r**2
        e = expected.copy()
        e[[4, 4, 5, 5], [4, 5, 4, 5]] = True
        simulator = CFDSimulator(p,v,b,k)
        simulator.scale_boundaries(scale=scale)
        nptest.assert_array_almost_equal(simulator.boundaries, e)
    
    def test_scale_down_field(self):
        u = self.zeros100.copy()
        simulator = CFDSimulator(self.zeros100, self.zeros1002, self.zeros100, 0.001) 

        u[11,11] = 5
        expected = self.zeros10.copy()
        expected[1,1] = 5
        res = simulator.scale_down_field(u)
        nptest.assert_allclose(res, expected)

    def test_scale_up_field(self):
        u = self.zeros10.copy()
        simulator = CFDSimulator(self.zeros100, self.zeros1002, self.zeros100, 0.001) 

        u[1,1] = 5
        expected = self.zeros100.copy()
        expected[:,:] = 1e-06
        expected[11,11] = 5
        res = simulator.scale_up_field(u)
        nptest.assert_allclose(res[11, 11], expected[11, 11], atol=1e-6)

