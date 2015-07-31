
from lib import Router
from lib.animators import *
from lib.simulators import *
from lib.draw import * 

from lib.solid_generators import * 

import sys
import numpy as np 

if sys.argv[1] == "--help":
    print("this width height velocity_x velocity_y num_iterations viscosity animator simulator domain")
    exit(0)

h = 1.0
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
animator_router.register(VelocityXAnimator, "velocity_x")
animator_router.register(VelocityYAnimator, "velocity_y")

simulator_router = Router()
simulator_router.register(LBMSimulator, "lbm")
simulator_router.register(CFDImplicitSimulator, "implicit")
simulator_router.register(CFDExplicitSimulator, "explicit")

domain_router = Router()
domain_router.register(blank, "blank")
domain_router.register(circle_center, "circle")
domain_router.register(square_center, "square")

animator_class = animator_router.route(sys.argv[7])
simulator_class = simulator_router.route(sys.argv[8])
domain_func = domain_router.route(sys.argv[9])

def v_func(x, y):
    return np.array([velocity_x, velocity_y])


p = np.zeros([n/h,m/h])
p.fill(10)

N,M = (int(n/h), int(m/h))

v = np.zeros([N,M, 2])
draw_from_function(v, N, M, v_func)
b = np.zeros([N,M])
draw_from_function(b,N,M, domain_func)

simulator = simulator_class(p,v,b, viscosity)
animator = animator_class(simulator) 
animator.run(num_iters, step=0.1)
