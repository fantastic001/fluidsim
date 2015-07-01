
class BaseSimulator(object):
    
    def __init__(self, boundary, viscosity):
        self.grid = boundary 
        self.viscosity = viscosity
        self.start()
    
    def start(self):
        pass
    
    def finish(self):
        pass

    def step(self, old, dt):
        """
        Must return new grid object 
        """
        pass

    def integrate(self, dt, T): 
        current = self.grid
        for i in range(int(T/dt)):
            current = self.step(current, dt)
        return current

