

from ..simulators import CFDSimulator, CFDImplicitSimulator, CFDExplicitSimulator
from ..structures import Cell 

from ..loggers import * 

import unittest 

import numpy as np
import numpy.testing as nptest


class TestCFDImplicitSimulator(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_advection(self):
        dt = 0.1
        size = 100
        h = 1/size
        # construct grid first 
        y, x = np.mgrid[0:1:h, 0:1:h]
        u = np.zeros([int(size), int(size), 2])
        expected = np.zeros([int(size),int(size), 2])
        
        n,m = size, size
        
        simulator = CFDImplicitSimulator(np.zeros([size,size]),np.zeros([size,size, 2]),np.zeros([size,size]),0, h=h)
        
        u[:,:,0] = 1*np.ones([n,m])
        u[:,:,1] = 1*np.ones([n,m])
        expected[:,:,0] = 1
        expected[:,:,1] = 1
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res, expected, rtol=0.05)
        self.assertTrue(np.floor(res.max()) <= np.floor(u.max()))

        u[:,:,0] = x/10
        u[:,:,1] = 0
        expected[:,:,0] = x/10 - dt*x/100
        expected[:,:,1] = 0
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res, expected, rtol=0.05)
        self.assertTrue(np.floor(res.max()) <= np.floor(u.max()))

        u[:,:,0] = x/10
        u[:,:,1] = y/10
        expected[:,:,0] = x/10 - dt*x/100
        expected[:,:,1] = y/10 - dt*y/100
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res, expected, rtol=0.05)
        self.assertTrue(np.floor(res.max()) <= np.floor(u.max()))

        u[:,:,0] = 1e-03*x*y
        u[:,:,1] = 1e-03*x*y
        expected[:,:,0] = 1e-03*x*y - 1e-06*dt*(x*y**2 + y*x**2)
        expected[:,:,1] = 1e-03*x*y - 1e-06*dt*(x*y**2 + y*x**2)
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res, expected, rtol=0.2)
        self.assertTrue(np.floor(res.max()) <= np.floor(u.max()))

        u[:,:,0] = 1e-03*(x**2 + y**2)
        u[:,:,1] = 1e-03*(x*y)
        expected[:,:,0] = 1e-03*(x**2 + y**2) - 1e-06*dt*(2*x*(x**2 + y**2) + 2*y*(x*y))
        expected[:,:,1] = 1e-03*x*y - 1e-06*dt*(y*(x**2 + y**2) + y*x**2)
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res, expected, rtol=0.2)
        self.assertTrue(np.floor(res.max()) <= np.floor(u.max()))
        
        # Symmetry test 
        u[:,:,0] = x/10
        u[:,:,1] = 0
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res[1:size // 2, 1:-1, 0], res[size-2:(size // 2)-1:-1, 1:-1, 0], atol=0.01, rtol=0.001)
        
        u[:,:,0] = x**2 / 10
        u[:,:,1] = 0
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res[1:size // 2, 1:-1, 0], res[size-2:(size // 2)-1:-1, 1:-1, 0], atol=0.01, rtol=0.001)
        
        u[:,:,0] = x**2 / 10
        u[:,:,1] = (0.5 - y) / 10
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res[1:size // 2, 1:-1, 0], res[size-2:(size // 2)-1:-1, 1:-1, 0], atol=0.01, rtol=0.001)
        

    def test_diffusion(self):
        k = 0.000001
        size = 100
        h = 1/size
        simulator = CFDImplicitSimulator(np.zeros([size,size]),np.zeros([size,size, 2]),np.zeros([size,size]),k, h=h)
        # construct grid first 
        y, x = np.mgrid[0:1:simulator.h, 0:1:simulator.h]
        u = np.zeros([int(size), int(size), 2])
        expected = np.zeros([int(size),int(size), 2])
        dt = 0.1

        u[:,:,0] = 1
        u[:,:,1] = 1
        res = simulator.diffusion(u, dt)
        expected[:,:,0] = 1
        expected[:,:,1] = 1
        nptest.assert_allclose(res[1:-1,1:-1], expected[1:-1,1:-1], atol=0.001, rtol=0.01)

        u[:,:,0] = x/10
        u[:,:,1] = y/10
        res = simulator.diffusion(u, dt)
        expected[:,:,0] = x/10
        expected[:,:,1] = y/10
        nptest.assert_allclose(res[1:-1,1:-1], expected[1:-1,1:-1], atol=0.001, rtol=0.01)

        u[:,:,0] = x**2 / 10
        u[:,:,1] = y**2 / 10
        res = simulator.diffusion(u, dt)
        expected[:,:,0] = x**2/10 + k*2*dt/10
        expected[:,:,1] = y**2/10 + k*2*dt/10
        nptest.assert_allclose(res[1:-1,1:-1], expected[1:-1,1:-1], atol=0.001, rtol=0.01)
        
        u[:,:,0] = (x**2 + y**2) / 10
        u[:,:,1] = (x**2 + y**2) / 10
        res = simulator.diffusion(u, dt)
        expected[:,:,0] = (x**2 + y**2)/10 + k*4*dt/10
        expected[:,:,1] = (x**2 + y**2)/10 + k*4*dt/10
        nptest.assert_allclose(res[1:-1,1:-1], expected[1:-1,1:-1], atol=0.001, rtol=0.01)
        
        # Symmetry test 
        u[:,:,0] = x/10
        u[:,:,1] = 0
        res = simulator.diffusion(u, dt)
        nptest.assert_allclose(res[1:size // 2, 1:-1, 0], res[size-2:(size // 2)-1:-1, 1:-1, 0], atol=0.01, rtol=0.001)
        
        u[:,:,0] = x**2/10 
        u[:,:,1] = 0
        res = simulator.diffusion(u, dt)
        nptest.assert_allclose(res[1:size // 2, 1:-1, 0], res[size-2:(size // 2)-1:-1, 1:-1, 0], atol=0.01, rtol=0.001)
        
        u[:,:,0] = x**2 /10
        u[:,:,1] = 0.5 - y/10
        res = simulator.diffusion(u, dt)
        nptest.assert_allclose(res[1:size // 2, 1:-1, 0], res[size-2:(size // 2)-1:-1, 1:-1, 0], atol=0.01, rtol=0.001)
