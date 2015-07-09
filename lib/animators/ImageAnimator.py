
from .BaseAnimator import * 

import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np 
class ImageAnimator(BaseAnimator):
    def start(self, simulator):
        self.fig = plt.figure()
        self.imas = []

    def update(self, p, v, b, t):
        n,m = p.shape
        maxDensity = 10.
        mat = np.zeros([n,m,3])
        mat[:,:,1] = b
        mat[:,:,2] = np.clip(p, 0, maxDensity) / maxDensity
        mat[:,:,2] *= np.ones(b.shape) - b # do not color boundaries to blue 
        plot = plt.imshow(mat)
        self.imas.append([plot])

    def finish(self):
        ani = animation.ArtistAnimation(self.fig, self.imas, interval=50, blit=True,repeat_delay=1000)
        plt.show()
