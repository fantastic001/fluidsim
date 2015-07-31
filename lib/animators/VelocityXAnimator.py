
from .BaseAnimator import * 

import matplotlib.pyplot as plt 
import numpy as np 
class VelocityXAnimator(BaseAnimator):
    def start(self, simulator, **kwargs):
        self.maximum = int(kwargs.get("maximum", 20))
        self.target_path = kwargs.get("target_path", "frames/")

    def update(self, p, v, b, t):
        print("Iteration %d" % t)
        n,m = p.shape
        nb = np.ones(b.shape) - b
        speed = v[:,:,0]
        mat = np.zeros([n,m,3])
        mat[:,:,0] = np.ones([n,m])*nb
        mat[:,:,1] = np.ones([n,m])*nb
        mat[:,:,1] = np.ones([n,m]) - np.clip(speed, 0, self.maximum) / self.maximum
        plt.imsave(self.target_path + "/figure-" + str(t) + ".png", mat)
