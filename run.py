
from lib.animators import *
from lib.simulators import LBMSimulator 
from lib.structures import Cell 
from lib.draw import * 

import numpy as np 

n,m = (100, 100)

cell = Cell(5.0, [0., 0.], False)

def v(x,y):
    if (x-50)**2 + (y-50)**2 <= 20**2:
        return np.array([x,y])
    else: 
        return np.array([0,0])

def b(x,y):
    if (x - 50)**2 + (y-50)**2 <= 15**2 or (x==10 or x==90 or y==10 or y==90):
        return True
    return False

grid = cell.make_grid(n, n)
boundary_draw(grid, n, m, b)
velocity_draw(grid, n, m, v)
border_draw(grid,n,m)

simulator = LBMSimulator(grid, 3.5)
animator = ImageAnimator(simulator) 
#debug = DebugAnimator(simulator)
animator.run(100)
#debug.run(100)
