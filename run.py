
from lib.animators import *
from lib.simulators import LBMSimulator 

import numpy as np 

n,m = (300, 300)

def v_func(x, y):
    if (x-m/2)**2 + (y-n/2)**2 <= (n/8)**2:
        return np.array([0.4, 0.5])
    else: 
        return np.array([0,0])

def b_func(x,y):
    if x == 0 or x == m or y == 0 or y==n:
        return True
    if (x - m/2)**2 + (y-n/2)**2 <= (n/16)**2 or (x==int(0.1*m) or x==int(0.9*m) or y==int(0.1*n) or y==int(0.9*n)):
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
