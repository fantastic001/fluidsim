

from ..simulators import LBMSimulator 
from ..structures import Cell 

import unittest 

import numpy as np
import numpy.testing as nptest


class TestLBMSimulator(unittest.TestCase):
    
    def setUp(self):
        self.boundaries = np.zeros([100, 100])
        self.boundaries.fill(False)
        self.velocities = np.zeros([100, 100, 2])
        self.densities = np.zeros([100, 100])
        self.densities.fill(1.0)
        self.simulator = LBMSimulator(self.densities, self.velocities, self.boundaries, 3.5)

    def test_step(self):
        self.simulator.step(1.0)
        p,v,b = self.simulator.data()
        # we have no motion, so we expect same situation after step 
        nptest.assert_array_almost_equal(p, self.densities)
        nptest.assert_array_almost_equal(v, self.velocities)
    
    def test_boundaries(self):
        # test with boundaries 
        for i in range(50, 60):
            self.simulator.boundaries[i][50] = True
            self.simulator.boundaries[i][60] = True
        for j in range(50, 61):
            self.simulator.boundaries[50][j] = True
            self.simulator.boundaries[59][j] = True
        self.simulator.velocities[49][50] = np.array([5, 5])
        self.simulator.start()
        self.simulator.step(1.0)
        p,v,b = self.simulator.data()
        nptest.assert_array_almost_equal(self.simulator.densities[51][50], p[51][50])
        self.simulator.finish()

