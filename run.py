
from lib import Router
from lib.animators import *
from lib.simulators import LBMSimulator 

import sys
import numpy as np 

if sys.argv[1] == "--help":
    print("this width height velocity_x velocity_y num_iterations viscosity animator")
    exit(0)

m,n = int(sys.argv[1]), int(sys.argv[2])
velocity_x, velocity_y = float(sys.argv[3]), float(sys.argv[4])
num_iters = int(sys.argv[5])
viscosity = float(sys.argv[6])

animator_router = Router()
animator_router.register(ImageAnimator, "density")
animator_router.register(DebugAnimator, "debug")
animator_router.register(SpeedAnimator, "speed")
animator_router.register(WaveAnimator, "waves")

animator_class = animator_router.route(sys.argv[7])

def v_func(x, y):
    if (x-m/2)**2 + (y-n/2)**2 <= 25**2:
        return np.array([velocity_x, velocity_y])
    else: 
        return np.array([0,0])

def b_func(x,y):
    if x == 0 or x == m or y == 0 or y==n:
        return True
    if (x - m/2)**2 + (y-n/2)**2 <= 8**2 or (x==int(0.1*m) or x==int(0.9*m) or y==int(0.1*n) or y==int(0.9*n)):
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

simulator = LBMSimulator(p,v,b, viscosity)
animator = animator_class(simulator) 
animator.run(num_iters)
