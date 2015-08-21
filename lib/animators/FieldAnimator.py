
from .BaseAnimator import * 

import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np 

import os 
import os.path 

class FieldAnimator(BaseAnimator):
    def start(self, simulator, **kwargs):
        self.target_path = kwargs.get("target_path", "frames/")
        self.scale = kwargs.get("scale", 100)

        if not os.path.isdir(self.target_path):
            os.mkdir(self.target_path)

    def update(self, p, v, b, t):
        print("Iteration: %d" % t) 
        n,m = p.shape
        f = 100*v
        mat = np.zeros([n,m,3])
        plt.quiver(v[::11,::11, 0], v[::11,::11,1], scale=self.scale)
        plt.savefig(self.target_path + "/field-%d.png" % t)
        plt.clf()

    def finish(self):
        pass
