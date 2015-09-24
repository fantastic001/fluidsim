
from ..simulators import BaseSimulator 

import unittest 

import numpy as np
import numpy.testing as nptest

class SimpleSimulator(BaseSimulator):
   
    def start(self):
        self.started = True 

    def finish(self):
        pass

    def step(self, dt):
        n,m = self.densities.shape
        for i in range(n):
            for j in range(m):
                self.densities[i][j] += dt

class TestBaseSimulator(unittest.TestCase):
    
    def setUp(self):
        self.p = np.zeros([2,2])
        self.simulator = SimpleSimulator(self.p, np.zeros([2, 2, 2]), np.zeros([2,2]), np.zeros([2,2, 2]), 1.0)

    def test_start(self):
        self.assertTrue(self.simulator.started)

    def test_step(self):
        self.simulator.step(1.0)
        res, v, b = self.simulator.data()
        nptest.assert_array_almost_equal(np.ones([2,2]), res)
        self.simulator.finish()

