
from .BaseAnimator import * 

import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import numpy as np 
class ImageAnimator(BaseAnimator):
    def start(self, simulator, **kwargs):
        self.fig = plt.figure()
        self.imas = []
        self.maxDensity = int(kwargs.get("max_density", 20))

    def update(self, p, v, b, t):
        n,m = p.shape
        mat = np.zeros([n,m,3])
        mat[:,:,0] = b
        mat[:,:,2] = 0.75
        mat[:,:,1] = np.clip(p, 0, self.maxDensity) / self.maxDensity
        
        # do not color boundaries
        mat[:,:,2] *= np.ones(b.shape) - b 
        mat[:,:,1] *= np.ones(b.shape) - b

        plot = plt.imshow(mat)
        self.imas.append([plot])

    def finish(self):
        ani = animation.ArtistAnimation(self.fig, self.imas, interval=50, blit=True,repeat_delay=100)
        plt.show()
