
import unittest 

import numpy as np
import numpy.testing as nptest

from ..utils import staggered

class TestStaggered(unittest.TestCase):
    
    def setUp(self):
        n = 100
        self.n = n
        self.normal_constant_u = np.ones([n,n])*100
        self.normal_constant_v = np.ones([n,n])*100

        self.staggered_constant_u = np.ones([n-1, n])*100
        self.staggered_constant_v = np.ones([n, n-1])*100

    def test_field_transpose(self):
        u,v = staggered.field_transpose(self.normal_constant_u, self.normal_constant_v)
        nptest.assert_allclose(u, self.normal_constant_u)
        nptest.assert_allclose(v, self.normal_constant_v)

    def test_to_centered(self):
        #ubc, vbc = staggered.attach_boundaries(self.staggered_constant_u, self.staggered_constant_v)
        ubc, vbc = self.staggered_constant_u, self.staggered_constant_v

        u,v = staggered.to_centered(ubc, vbc)
        nptest.assert_allclose(u[1:-1,1:-1], self.normal_constant_u[1:-1,1:-1])
        nptest.assert_allclose(v[1:-1,1:-1], self.normal_constant_v[1:-1,1:-1])
    
    def test_to_staggered(self):
        u,v = staggered.to_staggered(self.normal_constant_u, self.normal_constant_v)
        nptest.assert_allclose(u[1:-1,1:-1], self.staggered_constant_u[1:-1,1:-1])
        nptest.assert_allclose(v[1:-1,1:-1], self.staggered_constant_v[1:-1,1:-1])

    def test_transition(self):
        u,v = staggered.to_staggered(self.normal_constant_u, self.normal_constant_v)
        #ubc, vbc = staggered.attach_boundaries(u, v)
        ubc, vbc = u,v
        u,v = staggered.to_centered(ubc, vbc)

        nptest.assert_allclose(u[1:-1,1:-1], self.normal_constant_u[1:-1,1:-1])
        nptest.assert_allclose(v[1:-1,1:-1], self.normal_constant_v[1:-1,1:-1])
