
from .BaseAnimator import * 

import matplotlib.pyplot as plt 
import numpy as np 
class WaveAnimator(BaseAnimator):
    def start(self, simulator, **kwargs):
        plt.ylim(0, 10)
        plt.xlabel("Iteration (discrete time)")
        plt.ylabel("Difference between max and min density in the grid")
        self.diffs = []

    def update(self, p, v, b, t):
        diff = p.max() - p.min()
        self.diffs.append(diff)

    def finish(self):
        plt.plot(range(len(self.diffs)), self.diffs)
        plt.show()
