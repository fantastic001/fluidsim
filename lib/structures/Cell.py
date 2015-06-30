
import numpy as np 

class Cell(object):
    density = 0.0
    velocity = np.array([0.0, 0.0])
    solid = False 

    def __init__(self, density, velocity = np.array([0.0,0.0]), solid = False):
        self.density = density 
        self.solid = solid 
        self.velocity = np.array(velocity)

    def make_grid(self, n, m):
        """
        Makes matrix of specified size with cells as this one
        """
        mat = [] 
        for i in range(n):
            row = [] 
            for j in range(m):
                row.append(Cell(self.density, self.velocity, self.solid))
            mat.append(row)
        return mat

    def __eq__(self, other):
        return (self.density, self.solid) == (other.density, other.solid) and np.array_equal(self.velocity, other.velocity)

    def __ne__(self, other):
        return not self.__eq__(other)
