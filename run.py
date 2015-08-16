
from lib import Router
from lib.animators import *
from lib.simulators import *
from lib.draw import * 

from lib.solid_generators import * 
from lib.velocity_generators import * 

from lib.loggers import StandardOutputLogger 

import sys
import numpy as np 

import json 

if sys.argv[1] == "--help":
    print("Usage: %s params_path" % sys.argv[0])
    exit(0)

parameter_file = sys.argv[1]
f = open(parameter_file)
params = json.loads(f.read())
f.close()


h = 1.0
m,n = int(params.get("height", 100)), int(params.get("width", 100))
num_iters = int(params.get("iterations", 100))
viscosity = float(params.get("viscosity", 0.000001))

animator_router = Router()
animator_router.register(ImageAnimator, "density")
animator_router.register(DebugAnimator, "debug")
animator_router.register(SpeedAnimator, "speed")
animator_router.register(WaveAnimator, "waves")
animator_router.register(FieldAnimator, "field") 
animator_router.register(VelocityXAnimator, "velocity_x")
animator_router.register(VelocityYAnimator, "velocity_y")
animator_router.register(DivergenceAnimator, "divergence")

simulator_router = Router()
simulator_router.register(LBMSimulator, "lbm")
simulator_router.register(CFDImplicitSimulator, "implicit")
simulator_router.register(CFDExplicitSimulator, "explicit")

domain_router = Router()
domain_router.register(blank, "blank")
domain_router.register(circle_center, "circle")
domain_router.register(square_center, "square")

velocity_router = Router()
velocity_router.register(linear_velocity, "linear")
velocity_router.register(half_linear_velocity, "half_linear")
velocity_router.register(linear_velocity_small, "linear_small")
velocity_router.register(half_linear_velocity_small, "half_linear_small")
velocity_router.register(constant_velocity, "constant")
velocity_router.register(constant_velocity_small, "constant_small")
velocity_router.register(half_constant_velocity_small, "half_constant_small")
velocity_router.register(half_constant_velocity, "half_constant")
animator_params = params.get("animator_params", {})

animator_class = animator_router.route(params.get("animator", "speed"))
simulator_class = simulator_router.route(params.get("simulator", "implicit"))
domain_func = domain_router.route(params.get("domain", "blank"))
v_func = velocity_router.route(params.get("velocity", "linear"))


p = np.zeros([n/h,m/h])
p.fill(10)

N,M = (int(n/h), int(m/h))

v = np.zeros([N,M, 2])
draw_from_function(v, N, M, v_func)
b = np.zeros([N,M])
draw_from_function(b,N,M, domain_func)

simulator = simulator_class(p,v,b, viscosity, logger=StandardOutputLogger(debug=True))
animator = animator_class(simulator, **animator_params) 
animator.run(num_iters, step=0.1)
