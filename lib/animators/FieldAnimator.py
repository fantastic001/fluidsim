
from .BaseAnimator import * 

import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np 
class FieldAnimator(BaseAnimator):
    def start(self, simulator):
        pass

    def update(self, p, v, b, t):
        print("Iteration: %d" % t) 
        n,m = p.shape
        maxDensity = 10.
        mat = np.zeros([n,m,3])
        plt.quiver(v[::20,::20,0], v[::20,::20,1], units="width")
        plt.savefig("frames/field-%d.png" % t)
        plt.clf()

    def finish(self):
        pass
