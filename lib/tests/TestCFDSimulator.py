
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
        simulator = CFDSimulator(0,0,0,0) # we do not need to specify fluid properties for this test
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
        simulator = CFDSimulator(0,0,0,0)
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
        simulator = CFDSimulator(0,0,0,0)
        res = simulator.compute_laplacian(f, 0.1, 0.1)
        nptest.assert_array_almost_equal(expected, res, decimal=3)
