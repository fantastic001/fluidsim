
from ..simulators import CFDSimulator, CFDImplicitSimulator, CFDExplicitSimulator
from ..structures import Cell 

import unittest 

import numpy as np
import numpy.testing as nptest


class TestCFDExplicitSimulator(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_advection(self):
        simulator = CFDExplicitSimulator(np.zeros([10,10]),np.zeros([10,10, 2]),np.zeros([10,10]), np.zeros([10,10,2]),0)
        # construct grid first 
        y, x = np.mgrid[0:10:simulator.h, 0:10:simulator.h]
        u = np.zeros([int(10/simulator.h), int(10/simulator.h), 2])
        expected = np.zeros([int(10/simulator.h),int(10/simulator.h), 2])
        
        n,m = 10/simulator.h, 10/simulator.h
        # testing primitive advection first 
        u[:,:,0] = 100*np.ones([n,m])
        u[:,:,1] = 100*np.ones([n,m])
        res = simulator.advection_primitive(u)
        nptest.assert_array_almost_equal(
            res[1:-1,1:-1], 
            expected[1:-1,1:-1]
        )

        u[:,:,0] = x
        u[:,:,1] = y
        expected[:,:,0] = x
        expected[:,:,1] = y
        res = simulator.advection_primitive(u)
        nptest.assert_array_almost_equal(
            res[1:-1,1:-1], 
            expected[1:-1,1:-1]
        )

        u[:,:,0] = x*y
        u[:,:,1] = x*y
        expected[:,:,0] = x*y**2 + y*x**2
        expected[:,:,1] = x*y**2 + y*x**2
        res = simulator.advection_primitive(u)
        nptest.assert_allclose(
            res[1:-1,1:-1], 
            expected[1:-1,1:-1], 
            rtol=0.2
        )

        u[:,:,0] = x**2 + y**2
        u[:,:,1] = x*y
        expected[:,:,0] = 2*x*(x**2 + y**2) + 2*y*(x*y) 
        expected[:,:,1] = y*(x**2 + y**2) + y*x**2
        res = simulator.advection_primitive(u)
        nptest.assert_allclose(
            res[1:-1,1:-1], 
            expected[1:-1,1:-1], 
            rtol=0.2
        )
