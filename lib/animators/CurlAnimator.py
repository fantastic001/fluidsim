

from .BaseAnimator import * 

import matplotlib.pyplot as plt 
import numpy as np 

import os 
import os.path

class CurlAnimator(BaseAnimator):
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
        nb = np.ones(b.shape) - b
        i,j = np.mgrid[1:n-1, 1:m-1]
        curl = np.zeros([n,m])
        curl[i,j] = np.abs((v[i+1,j,0] - v[i-1,j,0]) - (v[i,j+1,1] - v[i,j+1,1]))
        mat = np.zeros([n,m,3])
        mat[:,:,0] = np.ones([n,m])*nb
        mat[:,:,1] = np.ones([n,m])*nb
        mat[:,:,1] = np.ones([n,m]) - np.clip(curl, 0, self.maximum) / self.maximum
        plt.imsave(self.target_path + "/figure-" + str(t) + ".png", mat)
        self.f_avg.write("%d,%f\n" % (t, curl.mean()))
        self.f_max.write("%d,%f\n" % (t, curl.max()))
