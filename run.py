
from lib import Router
from lib.animators import *
from lib.simulators import LBMSimulator 
from lib.draw import * 

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
animator_router.register(FieldAnimator, "field") 

animator_class = animator_router.route(sys.argv[7])

def v_func(x, y):
    #if (x-m/2)**2 + (y-n/2)**2 <= 25**2:
    if x > 2 and x < m-2 and y>2 and y<n-2:
        return np.array([velocity_x, velocity_y])
    else: 
        return np.array([0,0])

def b_func(x,y):
    if x == 2 or x == m-2 or y == 2 or y==n-2:
        return 1.0
    #if (x - m/2)**2 + (y-n/2)**2 <= 8**2 or (x==int(0.1*m) or x==int(0.9*m) or y==int(0.1*n) or y==int(0.9*n)):
    #    return True
    return 0.0

p = np.zeros([n,m])
p.fill(5.0)

v = np.zeros([n,m, 2])
draw_from_function(v, n, m, v_func)
b = np.zeros([n,m])
draw_from_function(b,n,m, b_func)

simulator = LBMSimulator(p,v,b, viscosity)
animator = animator_class(simulator) 
animator.run(num_iters)
