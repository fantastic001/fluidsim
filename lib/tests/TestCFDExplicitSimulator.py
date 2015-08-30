
from ..simulators import CFDSimulator, CFDImplicitSimulator, CFDExplicitSimulator
from ..structures import Cell 

import unittest 

import numpy as np
import numpy.testing as nptest


class TestCFDExplicitSimulator(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_advection(self):
        simulator = CFDExplicitSimulator(np.zeros([10,10]),np.zeros([10,10, 2]),np.zeros([10,10]),0, h=0.1)
        # construct grid first 
        y, x = np.mgrid[0:1:simulator.h, 0:1:simulator.h]
        u = np.zeros([int(1/simulator.h), int(1/simulator.h), 2])
        expected = np.zeros([int(1/simulator.h),int(1/simulator.h), 2])
        
        n,m = 1/simulator.h, 1/simulator.h
        # testing primitive advection first 
        u[:,:,0] = 0.1*np.ones([n,m])
        u[:,:,1] = 0.1*np.ones([n,m])
        res = simulator.advection_primitive(u)
        nptest.assert_array_almost_equal(res, expected)

        u[:,:,0] = x/10
        u[:,:,1] = y/10
        expected[:,:,0] = x/100
        expected[:,:,1] = y/100
        res = simulator.advection_primitive(u)
        nptest.assert_array_almost_equal(res, expected)

        u[:,:,0] = x*y/10
        u[:,:,1] = x*y/10
        expected[:,:,0] = (x*y**2 + y*x**2) / 100
        expected[:,:,1] = (x*y**2 + y*x**2) / 100
        res = simulator.advection_primitive(u)
        nptest.assert_allclose(res, expected, rtol=0.2)

        u[:,:,0] = (x**2 + y**2) / 10
        u[:,:,1] = x*y / 10
        expected[:,:,0] = (2*x*(x**2 + y**2) + 2*y*(x*y) ) / 100
        expected[:,:,1] = (y*(x**2 + y**2) + y*x**2) / 100
        res = simulator.advection_primitive(u)
        nptest.assert_allclose(res, expected, rtol=0.2)
