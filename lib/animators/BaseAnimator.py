
class BaseAnimator(object):
    
    def __init__(self, simulator):
        self.simulator = simulator 
        self.start(self.simulator)
    
    def start(self, simulator):
        pass

    def update(self, grid, iteration):
        pass 

    def run(self, iters, step=0.1):
        current = self.simulator.grid
        self.update(current, 0)
        for i in range(iters):
            current = self.simulator.step(current, step)
            self.update(current, i+1)
