

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
        
        # test with boundaries 
        fluid_cell = Cell(1.0, [0,0], False)
        mat = fluid_cell.make_grid(100, 100)
        for i in range(50, 60):
            mat[i][50].solid = True
            mat[i][60].solid = True
        for j in range(50, 61):
            mat[50][j].solid = True
            mat[59][j].solid = True
        mat[49][50].velocity = np.array([5, 5])
        res = self.simulator.step(mat, 1.0)
        self.assertEqual(mat[51][50].density, res[51][50].density)

        self.simulator.finish()

    def test_integration(self):
        # same here ....
        res = self.simulator.integrate(0.1, 1)
        self.assertEqual(res, res)
        self.simulator.finish()

