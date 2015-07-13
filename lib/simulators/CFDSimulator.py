
import numpy as np

from .BaseSimulator import BaseSimulator

class CFDSimulator(BaseSimulator):

    def compute_gradient(self, a, dx, dy):
        dy, dx = np.gradient(a, dy, dx)
        grad = np.zeros([a.shape[0], a.shape[1], 2])
        grad[:,:,0] = dx 
        grad[:,:,1] = dy
        return grad

    def compute_divergence(self, f, dx, dy):
        p = f[:,:,0]
        q = f[:,:,1]
        dp = self.compute_gradient(p, dx, dy) 
        dq = self.compute_gradient(q, dx, dy)
        dpdx = dp[:,:,0]
        dqdy = dq[:,:,1]
        return dpdx + dqdy

    def start(self):
        pass
    
    def finish(self):
        pass

    def step(self, dt):
        """
        Must return new grid object 
        """
        pass
