
from ..simulators import BaseSimulator 

import unittest 

import numpy as np
import numpy.testing as nptest

class SimpleSimulator(BaseSimulator):
   
    def start(self):
        self.started = True 

    def finish(self):
        pass

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
        self.simulator = SimpleSimulator(self.grid, 1.0)

    def test_start(self):
        self.assertTrue(self.simulator.started)

    def test_step(self):
        res = self.simulator.step(self.grid, 1.0)
        nptest.assert_array_almost_equal(np.ones([2, 2]), res)
        self.simulator.finish()

    def test_integration(self):
        res = self.simulator.integrate(0.1, 1)
        nptest.assert_array_almost_equal(res, np.ones([2, 2]))
        self.simulator.finish()
