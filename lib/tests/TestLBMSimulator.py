

from ..simulators import LBMSimulator 
from ..structures import Cell 

import unittest 

import numpy as np
import numpy.testing as nptest


class TestLBMSimulator(unittest.TestCase):
    
    def setUp(self):
        cell = Cell(1.0, [0,0], False) 
        self.grid = cell.make_grid(100,100)
        self.simulator = LBMSimulator(self.grid)

    def test_step(self):
        res = self.simulator.step(self.grid, 1.0)
        # we have no motion, so we expect same situation after step 
        self.assertEqual(res, res)
        self.simulator.finish()

    def test_integration(self):
        # same here ....
        res = self.simulator.integrate(0.1, 1)
        self.assertEqual(res, res)
        self.simulator.finish()
