

from ..simulators import CFDSimulator, CFDImplicitSimulator, CFDExplicitSimulator
from ..structures import Cell 

import unittest 

import numpy as np
import numpy.testing as nptest


class TestCFDImplicitSimulator(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_diffusion(self):
        k = 0.000001
        size = 100
        simulator = CFDImplicitSimulator(np.zeros([size,size]),np.zeros([size,size, 2]),np.zeros([size,size]),k)
        # construct grid first 
        y, x = np.mgrid[0:size:simulator.h, 0:size:simulator.h]
        u = np.zeros([int(size/simulator.h), int(size/simulator.h), 2])
        expected = np.zeros([int(size/simulator.h),int(size/simulator.h), 2])
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
