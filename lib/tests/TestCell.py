
from ..structures import Cell

import unittest 

import numpy as np
import numpy.testing as nptest


class TestCell(unittest.TestCase):
    
    def test_cell_init(self):
        cell = Cell(1.0)
        self.assertEqual(1.0, cell.density)
        cell = Cell(1.0, [0,0], True)
        self.assertEqual(1.0, cell.density)
        self.assertEqual(True, cell.solid)
        nptest.array_almost_equal(cell.velocity, [0,0])

    def test_make_grid(self):
        cell = Cell(1.0)
        res = cell.make_grid(5,5)
        self.assertEqual(res.shape (5,5))
