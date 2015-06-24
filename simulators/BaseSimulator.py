
class BaseSimulator(object):
    
    def __init__(self, boundary):
        self.grid = boundary 

    def step(self, old, dt):
        """
        Must return new grid object 
        """
        pass

    def integrate(self, dt, T): 
        current = self.boundary
        for i in range(T/dt):
            current = self.step(current, dt)

