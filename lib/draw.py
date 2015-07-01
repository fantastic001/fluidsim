
def boundary_draw(grid, n, m, func):
    """
    Draws boundary 

    if func(x,y) is True, then it will be solid, fluid otherwise 
    """
    for i in range(n):
        for j in range(m):
            grid[i][j].solid = func(j,i)


def density_draw(grid, n, m, func):
    """
    Draws densities

    density(x,y) = func(x,y)
    """
    for i in range(n):
        for j in range(m):
            grid[i][j].density = func(j,i)

def velocity_draw(grid, n, m, func):
    """
    Draws velocities

    velocity(x,y) = func(x,y)
    """
    for i in range(n):
        for j in range(m):
            grid[i][j].velocity = func(j,i)
