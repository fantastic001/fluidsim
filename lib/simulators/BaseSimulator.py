
class BaseSimulator(object):
    
    def __init__(self, densities, velocities, boundary, viscosity):
        self.boundaries = boundary 
        self.viscosity = viscosity
        self.densities = densities 
        self.velocities = velocities
        self.start()
    
    def start(self):
        pass
    
    def finish(self):
        pass

    def step(self, dt):
        """
        Must return new grid object 
        """
        pass

    def data(self):
        return (self.densities, self.velocities, self.boundaries)
