
from ..loggers import BaseLogger

class BaseSimulator(object):
    
    def __init__(self, densities, velocities, boundary, viscosity, logger=None, **kwargs):
        self.boundaries = boundary 
        self.viscosity = viscosity
        self.densities = densities 
        self.velocities = velocities
        if logger != None:
            self.logger = logger 
        else:
            self.logger = BaseLogger()
        self.start(**kwargs)
    
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
