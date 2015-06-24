
from ..simulators import BaseSimulator 

import unittest 

import numpy as np
import numpy.testing as nptest

class SimpleSimulator(BaseSimulator):
    
    def step(self, old, dt):
        n,m = old.shape 
        new = np.zeros([n,m])
        for i in range(n):
            for j in range(m):
                new[i][j] = old[i][j] + dt
        return new 

class TestBaseSimulator(unittest.TestCase):
    
    def setUp(self):
        self.grid = np.zeros([2,2])
        self.simulator = SimpleSimulator(self.grid)

    def test_step(self):
        res = self.simulator.step(self.grid, 1.0)
        nptest.assert_array_almost_equal(np.ones([2, 2]), res)

    def test_integration(self):
        res = self.simulator.integrate(0.1, 1)
        nptest.assert_array_almost_equal(res, np.ones([2, 2]))
