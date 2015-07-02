
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

def border_draw(grid,n,m):
    """
    Draws solid borders
    """
    for i in range(n):
        for j in range(m):
            if i == 0 or j == 0 or i == n-1 or j == m-1:
                grid[i][j].solid = True
