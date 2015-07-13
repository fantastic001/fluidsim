
import numpy as np

from .BaseSimulator import BaseSimulator

class CFDSimulator(BaseSimulator):

    def compute_gradient(self, a, dx, dy):
        dy, dx = np.gradient(a, dy, dx)
        grad = np.zeros([a.shape[0], a.shape[1], 2])
        grad[:,:,0] = dx 
        grad[:,:,1] = dy
        return grad

    def start(self):
        pass
    
    def finish(self):
        pass

    def step(self, dt):
        """
        Must return new grid object 
        """
        pass
