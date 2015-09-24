
from ..loggers import BaseLogger

class BaseSimulator(object):
    
    def __init__(self, densities, velocities, boundary, forces, viscosity, logger=None):
        self.boundaries = boundary 
        self.viscosity = viscosity
        self.densities = densities 
        self.velocities = velocities
        self.forces = forces
        if logger != None:
            self.logger = logger 
        else:
            self.logger = BaseLogger()
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
