
from .BaseAnimator import * 

import matplotlib.pyplot as plt 
import numpy as np 
class ImageAnimator(BaseAnimator):
    maxDensity = 10.
    def start(self, simulator):
        pass

    def update(self, p, v, b, t):
        n,m = p.shape
        mat = np.zeros([n,m,3])
        for i in range(n):
            for j in range(m):
                if b[i][j]:
                    mat[i][j][1] = 1.0
                else:
                    mat[i][j][2] = p[i][j] / self.maxDensity 
                    if mat[i][j][2] > self.maxDensity:
                        mat[i][j][2] = self.maxDensity
        plt.imsave("figure-" + str(t) + ".png", mat)
