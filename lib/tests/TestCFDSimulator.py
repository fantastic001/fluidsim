
from ..simulators import CFDSimulator
from ..structures import Cell 

import unittest 

import numpy as np
import numpy.testing as nptest


class TestCFDSimulator(unittest.TestCase):
    
    def setUp(self):
        """
        self.boundaries = np.zeros([100, 100])
        self.boundaries.fill(False)
        self.velocities = np.zeros([100, 100, 2])
        self.densities = np.zeros([100, 100])
        self.densities.fill(1.0)
        self.simulator = LBMSimulator(self.densities, self.velocities, self.boundaries, 3.5)
        """

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

    def test_divergence(self):
        y,x = np.mgrid[0:100:0.1, 0:100:0.1]
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

    def test_laplacian(self):
        y,x = np.mgrid[0:100:0.1, 0:100:0.1]
        f = np.zeros([x.shape[0], x.shape[1], 2])
        f[:,:,0] = x**2
        f[:,:,1] = y**2
        expected = 2*np.ones([f.shape[0], f.shape[1], 2])
        simulator = CFDSimulator(np.zeros([100,100]),np.zeros([100,100,2]),np.zeros([100,100]),0)
        res = simulator.compute_laplacian(f, 0.1, 0.1)
        nptest.assert_array_almost_equal(expected, res, decimal=3)

    def test_advection(self):
        simulator = CFDSimulator(np.zeros([10,10]),np.zeros([10,10, 2]),np.zeros([10,10]),0)
        # construct grid first 
        y, x = np.mgrid[0:10:simulator.h, 0:10:simulator.h]
        u = np.zeros([int(10/simulator.h), int(10/simulator.h), 2])
        expected = np.zeros([int(10/simulator.h),int(10/simulator.h), 2])
        
        n,m = 10/simulator.h, 10/simulator.h
        # testing primitive advection first 
        u[:,:,0] = 100*np.ones([n,m])
        u[:,:,1] = 100*np.ones([n,m])
        res = simulator.advection_primitive(u)
        nptest.assert_array_almost_equal(res, expected)

        u[:,:,0] = x
        u[:,:,1] = y
        expected[:,:,0] = x
        expected[:,:,1] = y
        res = simulator.advection_primitive(u)
        nptest.assert_array_almost_equal(res, expected)

        u[:,:,0] = x*y
        u[:,:,1] = x*y
        expected[:,:,0] = 2*x*(y**2)
        expected[:,:,1] = 2*y*(x**2)
        res = simulator.advection_primitive(u)
        nptest.assert_allclose(res, expected, rtol=0.1)

    def test_diffusion(self):
        k = 0.000001
        simulator = CFDSimulator(np.zeros([100,100]),np.zeros([100,100, 2]),np.zeros([100,100]),k)
        # construct grid first 
        y, x = np.mgrid[0:100:simulator.h, 0:100:simulator.h]
        u = np.zeros([int(100/simulator.h), int(100/simulator.h), 2])
        expected = np.zeros([int(100/simulator.h),int(100/simulator.h), 2])
        dt = 0.1

        u[:,:,0] = 100
        u[:,:,1] = 100
        res = simulator.diffusion(u, dt)
        expected[:,:,0] = 100
        expected[:,:,1] = 100
        nptest.assert_array_almost_equal(res[1:-1,1:-1], expected[1:-1,1:-1], decimal=3)

        u[:,:,0] = x
        u[:,:,1] = y
        res = simulator.diffusion(u, dt)
        expected[:,:,0] = x
        expected[:,:,1] = y
        nptest.assert_array_almost_equal(res[1:-1,1:-1], expected[1:-1,1:-1], decimal=3)

        u[:,:,0] = x**2
        u[:,:,1] = y**2
        res = simulator.diffusion(u, dt)
        expected[:,:,0] = x**2 + k*2*dt
        expected[:,:,1] = y**2 + k*2*dt
        nptest.assert_array_almost_equal(res[1:-1,1:-1], expected[1:-1,1:-1], decimal=3)
        
        u[:,:,0] = x**2 + y**2
        u[:,:,1] = x**2 + y**2
        res = simulator.diffusion(u, dt)
        expected[:,:,0] = x**2 + y**2 + k*4*dt
        expected[:,:,1] = x**2 + y**2 + k*4*dt
        nptest.assert_array_almost_equal(res[1:-1,1:-1], expected[1:-1,1:-1], decimal=2)

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

    def test_projection(self):
        # WARNING: This test assumes correct compute_divergence method
        # TODO Fix this to be independent
        simulator = CFDSimulator(np.zeros([100,100]),np.zeros([100,100, 2]),np.zeros([100,100]),0.001)
        # construct grid first 
        y, x = np.mgrid[0:100:simulator.h, 0:100:simulator.h]
        u = np.zeros([int(100/simulator.h), int(100/simulator.h), 2])
        expected = np.zeros([int(100/simulator.h),int(100/simulator.h)])
        dt = 0.1

        u[:,:,0] = 1e+15
        u[:,:,1] = 1e+15
        w, p = simulator.projection(u, dt)
        nptest.assert_array_almost_equal(simulator.compute_divergence(w, simulator.h, simulator.h), expected)
        
        u[:,:,0] = x**3 + 2*x**2*y**3 + 4*y**2
        u[:,:,1] = 2*x**2 + y**2
        w, p = simulator.projection(u, dt)
        nptest.assert_array_almost_equal(simulator.compute_divergence(w, simulator.h, simulator.h), expected)
