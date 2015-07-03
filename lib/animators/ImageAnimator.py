
from .BaseAnimator import * 

import matplotlib.pyplot as plt 
import numpy as np 
class ImageAnimator(BaseAnimator):
    def start(self, simulator):
        pass

    def update(self, p, v, b, t):
        n,m = p.shape
        maxDensity = 10.
        mat = np.zeros([n,m,3])
        mat[:,:,1] = b
        mat[:,:,2] = np.clip(p, 0, maxDensity) / maxDensity
        mat[:,:,2] *= np.ones(b.shape) - b # do not color boundaries to blue 
        plt.imsave("frames/figure-" + str(t) + ".png", mat)
