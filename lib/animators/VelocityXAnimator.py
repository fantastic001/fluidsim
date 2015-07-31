
from .BaseAnimator import * 

import matplotlib.pyplot as plt 
import numpy as np 
class VelocityXAnimator(BaseAnimator):
    def start(self, simulator, **kwargs):
        pass

    def update(self, p, v, b, t):
        print("Iteration %d" % t)
        n,m = p.shape
        nb = np.ones(b.shape) - b
        maximum = 20
        speed = v[:,:,0]
        mat = np.zeros([n,m,3])
        mat[:,:,0] = np.ones([n,m])*nb
        mat[:,:,1] = np.ones([n,m])*nb
        mat[:,:,1] = np.ones([n,m]) - np.clip(speed, 0, maximum) / maximum
        plt.imsave("frames/figure-" + str(t) + ".png", mat)