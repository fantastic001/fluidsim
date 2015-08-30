

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
        # construct grid first 
        y, x = np.mgrid[0:size:1, 0:size:1]
        u = np.zeros([int(size), int(size), 2])
        expected = np.zeros([int(size),int(size), 2])
        
        n,m = size, size
        
        simulator = CFDImplicitSimulator(np.zeros([size,size]),np.zeros([size,size, 2]),np.zeros([size,size]),0)
        
        u[:,:,0] = 100*np.ones([n,m])
        u[:,:,1] = 100*np.ones([n,m])
        expected[:,:,0] = 100
        expected[:,:,1] = 100
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res, expected, rtol=0.05)
        self.assertTrue(np.floor(res.max()) <= np.floor(u.max()))

        u[:,:,0] = x
        u[:,:,1] = 0
        expected[:,:,0] = x - dt*x
        expected[:,:,1] = 0
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res, expected, rtol=0.05, atol=0.01)
        self.assertTrue(np.floor(res.max()) <= np.floor(u.max()))

        u[:,:,0] = x
        u[:,:,1] = y
        expected[:,:,0] = x - dt*x
        expected[:,:,1] = y - dt*y
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res, expected, rtol=0.05, atol=0.01)
        self.assertTrue(np.floor(res.max()) <= np.floor(u.max()))

        u[:,:,0] = 1e-03*x*y
        u[:,:,1] = 1e-03*x*y
        expected[:,:,0] = 1e-03*x*y - 1e-06*dt*(x*y**2 + y*x**2)
        expected[:,:,1] = 1e-03*x*y - 1e-06*dt*(x*y**2 + y*x**2)
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res, expected, rtol=0.2, atol=1)
        self.assertTrue(np.floor(res.max()) <= np.floor(u.max()))

        u[:,:,0] = 1e-03*(x**2 + y**2)
        u[:,:,1] = 1e-03*(x*y)
        expected[:,:,0] = 1e-03*(x**2 + y**2) - 1e-06*dt*(2*x*(x**2 + y**2) + 2*y*(x*y))
        expected[:,:,1] = 1e-03*x*y - 1e-06*dt*(y*(x**2 + y**2) + y*x**2)
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res, expected, rtol=0.2, atol=1)
        self.assertTrue(np.floor(res.max()) <= np.floor(u.max()))
        
        # Symmetry test 
        u[:,:,0] = x
        u[:,:,1] = 0
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res[1:size // 2, 1:-1, 0], res[size-2:(size // 2)-1:-1, 1:-1, 0], atol=0.01, rtol=0.001)
        
        u[:,:,0] = x**2 
        u[:,:,1] = 0
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res[1:size // 2, 1:-1, 0], res[size-2:(size // 2)-1:-1, 1:-1, 0], atol=0.01, rtol=0.001)
        
        u[:,:,0] = x**2 
        u[:,:,1] = size/2 - y
        res = simulator.perform_advection(u, dt)
        nptest.assert_allclose(res[1:size // 2, 1:-1, 0], res[size-2:(size // 2)-1:-1, 1:-1, 0], atol=0.01, rtol=0.001)
        

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
        nptest.assert_allclose(res[1:-1,1:-1], expected[1:-1,1:-1], atol=0.001, rtol=0.01)

        u[:,:,0] = x
        u[:,:,1] = y
        res = simulator.diffusion(u, dt)
        expected[:,:,0] = x
        expected[:,:,1] = y
        nptest.assert_allclose(res[1:-1,1:-1], expected[1:-1,1:-1], atol=0.001, rtol=0.01)

        u[:,:,0] = x**2
        u[:,:,1] = y**2
        res = simulator.diffusion(u, dt)
        expected[:,:,0] = x**2 + k*2*dt
        expected[:,:,1] = y**2 + k*2*dt
        nptest.assert_allclose(res[1:-1,1:-1], expected[1:-1,1:-1], atol=0.001, rtol=0.01)
        
        u[:,:,0] = x**2 + y**2
        u[:,:,1] = x**2 + y**2
        res = simulator.diffusion(u, dt)
        expected[:,:,0] = x**2 + y**2 + k*4*dt
        expected[:,:,1] = x**2 + y**2 + k*4*dt
        nptest.assert_allclose(res[1:-1,1:-1], expected[1:-1,1:-1], atol=0.001, rtol=0.01)
        
        # Symmetry test 
        u[:,:,0] = x
        u[:,:,1] = 0
        res = simulator.diffusion(u, dt)
        nptest.assert_allclose(res[1:size // 2, 1:-1, 0], res[size-2:(size // 2)-1:-1, 1:-1, 0], atol=0.01, rtol=0.001)
        
        u[:,:,0] = x**2 
        u[:,:,1] = 0
        res = simulator.diffusion(u, dt)
        nptest.assert_allclose(res[1:size // 2, 1:-1, 0], res[size-2:(size // 2)-1:-1, 1:-1, 0], atol=0.01, rtol=0.001)
        
        u[:,:,0] = x**2 
        u[:,:,1] = size/2 - y
        res = simulator.diffusion(u, dt)
        nptest.assert_allclose(res[1:size // 2, 1:-1, 0], res[size-2:(size // 2)-1:-1, 1:-1, 0], atol=0.01, rtol=0.001)

    
    def test_step_blank(dt):
        size = 100
        
        dt = 0.1
        k = 1e-06
        y,x = np.mgrid[0:size, 0:size]
        r = 5
        b = np.zeros([size, size]).astype(bool)
        v = np.zeros([size, size, 2])
        v[:, 0:0.4*size, 0] = np.linspace(size // 10, 0, 40)
        simulator = CFDImplicitSimulator(np.zeros([size, size]), v, b, k)

        # after 5 iterations, we have to have symmetry
        for iteration in range(5):
            simulator.step(dt)
        res = simulator.velocities
        nptest.assert_allclose(res[1:size // 2, 1:-1, 0], res[size-2:(size // 2)-1:-1, 1:-1, 0],
            err_msg="Not symmetric by row"
        )
        nptest.assert_allclose(res[1:size // 2, 1:-1, 1], -res[size-2:(size // 2)-1:-1, 1:-1, 1],
            err_msg="Not symmetric by column"
        )

    def test_step_circle(self):
        size = 100
        
        dt = 0.1
        k = 1e-06
        y,x = np.mgrid[0:size, 0:size]
        r = 5
        b = (x - 0.495*size)**2 + (y - 0.495*size)**2 <= r**2
        v = np.zeros([size, size, 2])
        v[:, 0:0.4*size, 0] = np.linspace(size // 10, 0, 40)
        simulator = CFDImplicitSimulator(np.zeros([size, size]), v, b, k)

        # after 5 iterations, we have to have symmetry
        for iteration in range(5):
            simulator.step(dt)
        res = simulator.velocities
        nptest.assert_allclose(res[1:size // 2, 1:-1, 0], res[size-2:(size // 2)-1:-1, 1:-1, 0],
            err_msg="Not symmetric by row"
        )
        nptest.assert_allclose(res[1:size // 2, 1:-1, 1], -res[size-2:(size // 2)-1:-1, 1:-1, 1],
            err_msg="Not symmetric by column"
        )
