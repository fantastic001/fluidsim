
import numpy as np

from .BaseSimulator import BaseSimulator

class CFDSimulatorSimulator(BaseSimulator):

    def compute_gradient(self, a, dx, dy):
        da = np.gradient(a, [dx, dy])
        return da 

    def start(self):
        pass
    
    def finish(self):
        pass

    def step(self, dt):
        """
        Must return new grid object 
        """
        pass
