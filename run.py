
from lib.animators import *
from lib.simulators import LBMSimulator 

import numpy as np 

n,m = (100, 100)

def v_func(x, y):
    if (x-50)**2 + (y-50)**2 <= 20**2:
        return np.array([x,y])
    else: 
        return np.array([0,0])

def b_func(x,y):
    if x == 0 or x == m or y == 0 or y==n:
        return True
    if (x - 50)**2 + (y-50)**2 <= 15**2 or (x==10 or x==90 or y==10 or y==90):
        return True
    return False

p = np.zeros([n,m])
p.fill(5.0)

v = np.zeros([n,m, 2])
for i in range(n):
    for j in range(m):
        v[i,j,:] = v_func(j,i)

b = np.zeros([n,m])
for i in range(n):
    for j in range(m):
        b[i,j] = b_func(j,i)

simulator = LBMSimulator(p,v,b, 3.5)
animator = ImageAnimator(simulator) 
#debug = DebugAnimator(simulator)
animator.run(100)
#debug.run(100)
