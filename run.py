
from lib.animators import ImageAnimator 
from lib.simulators import LBMSimulator 
from lib.structures import Cell 
from lib.draw import * 

import numpy as np 

n,m = (100, 100)

cell = Cell(5.0, [0., 0.], False)

grid = cell.make_grid(n, n)

boundary_draw(grid, n, m, lambda x,y: (x - 50)**2 + (y-50)**2 <= 15)
velocity_draw(grid, n, m, lambda x,y: np.array([x, y]))

simulator = LBMSimulator(grid, 3.5)
animator = ImageAnimator(simulator) 

animator.run(5)
