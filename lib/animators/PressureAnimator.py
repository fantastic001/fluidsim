

from .BaseAnimator import * 

import matplotlib.pyplot as plt 
import numpy as np 

import os 
import os.path

class PressureAnimator(BaseAnimator):
    def start(self, simulator, **kwargs):
        self.maximum = int(kwargs.get("maximum", 20))
        self.target_path = kwargs.get("target_path", "frames/")

        if not os.path.isdir(self.target_path):
            os.mkdir(self.target_path)

        self.f_avg = open(self.target_path + "/average-speed.csv", "w")
        self.f_max = open(self.target_path + "/maximum-speed.csv", "w")
    
    def update(self, p, v, b, t):
        print("Iteration %d" % t)
        n,m = p.shape
        P = self.simulator.pressure
        mat = np.zeros([n,m,3])
        mat[:,:,0] = P
        plt.imsave(self.target_path + "/figure-" + str(t) + ".png", mat, vmin=-self.maximum, vmax=self.maximum)
        self.f_avg.write("%d,%f\n" % (t, P.mean()))
        self.f_max.write("%d,%f\n" % (t, P.max()))
