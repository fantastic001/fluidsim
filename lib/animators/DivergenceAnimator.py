

from .BaseAnimator import * 

import matplotlib.pyplot as plt 
import numpy as np 

import os 
import os.path

class DivergenceAnimator(BaseAnimator):
    def start(self, simulator, **kwargs):
        self.target_path = kwargs.get("target_path", "frames/")

        if not os.path.isdir(self.target_path):
            os.mkdir(self.target_path)

        self.f = open(self.target_path + "/data.csv", "w")
    
    def compute_gradient(self, a, dx, dy, edge_order=2):
        dy, dx = np.gradient(a, dy, dx, edge_order=edge_order)
        grad = np.zeros([a.shape[0], a.shape[1], 2])
        grad[:,:,0] = dx 
        grad[:,:,1] = dy
        return grad

    def compute_divergence(self, f, dx, dy, edge_order=2):
        p = f[:,:,0]
        q = f[:,:,1]
        dp = self.compute_gradient(p, dx, dy, edge_order=edge_order) 
        dq = self.compute_gradient(q, dx, dy, edge_order=edge_order)
        dpdx = dp[:,:,0]
        dqdy = dq[:,:,1]
        return dpdx + dqdy
    
    def finish(self):
        self.f.close()

    def update(self, p, v, b, t):
        print("Iteration %d" % t)
        n,m = p.shape
        div = self.compute_divergence(v, 1.0, 1.0)
        self.f.write("%d,%d\n" % (t, div.max()))

