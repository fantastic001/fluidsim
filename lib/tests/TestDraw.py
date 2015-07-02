


from ..draw import * 
from ..structures import Cell 

import unittest 

import numpy as np
import numpy.testing as nptest


class TestDraw(unittest.TestCase):
    
    def setUp(self):
        self.cell = Cell(1.0, [0,0], False)
        self.grid = self.cell.make_grid(10, 10)

    def test_boundary_draw(self):
        boundary_draw(self.grid, 10, 10, lambda x,y: x==5 and y == 5)
        self.assertTrue(self.grid[5][5].solid)
        self.assertFalse(self.grid[4][4].solid)

    def test_velocity_draw(self):
        velocity_draw(self.grid, 10, 10, lambda x,y: np.array([x, y]))
        nptest.assert_array_almost_equal(self.grid[5][5].velocity, np.array([5, 5]))

    def test_density_draw(self):
        density_draw(self.grid, 10, 10, lambda x,y: x**2 + y**2)
        self.assertEqual(self.grid[5][5].density, 50)

    def test_border_draw(self):
        border_draw(self.grid,10,10)
        self.assertTrue(self.grid[0][0].solid)
        self.assertTrue(self.grid[0][1].solid)
        self.assertTrue(self.grid[1][0].solid)
        self.assertTrue(self.grid[9][9].solid)
        self.assertTrue(self.grid[9][2].solid)
