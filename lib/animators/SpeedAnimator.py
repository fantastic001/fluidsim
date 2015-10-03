

from .BaseAnimator import * 

import matplotlib.pyplot as plt 
import numpy as np 

import os 
import os.path

class SpeedAnimator(BaseAnimator):
    def start(self, simulator, **kwargs):
        self.maximum = float(kwargs.get("maximum", 20))
        self.target_path = kwargs.get("target_path", "frames/")

        if not os.path.isdir(self.target_path):
            os.mkdir(self.target_path)

        self.f_avg = open(self.target_path + "/average-speed.csv", "w")
        self.f_max = open(self.target_path + "/maximum-speed.csv", "w")
    
    def update(self, p, v, b, t):
        print("Iteration %d" % t)
        n,m = p.shape
        nb = np.ones(b.shape) - b
        speed = np.sqrt(v[:,:,0]**2 + v[:,:,1]**2)
        mat = np.zeros([n,m,3])
        mat[:,:,2] = np.ones([n,m])*nb
        mat[:,:,1] = np.ones([n,m])*nb
        mat[:,:,1] = np.clip(speed, 0, self.maximum) / self.maximum
        plt.imsave(self.target_path + "/figure-" + str(t) + ".png", mat)
        self.f_avg.write("%d,%f\n" % (t, speed.mean() / self.maximum))
        self.f_max.write("%d,%f\n" % (t, speed.max() / self.maximum))
