
from .BaseAnimator import * 

import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np 
class FieldAnimator(BaseAnimator):
    def start(self, simulator, **kwargs):
        pass

    def update(self, p, v, b, t):
        print("Iteration: %d" % t) 
        n,m = p.shape
        mat = np.zeros([n,m,3])
        plt.quiver(v[::10,::10, 0], v[::10,::10,1], units="width")
        plt.savefig("frames/field-%d.png" % t)
        plt.clf()

    def finish(self):
        pass
