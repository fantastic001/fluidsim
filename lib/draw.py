
def draw_from_function(grid, n, m, func):
    """
    Draws boundary 

    if func(x,y) is True, then it will be solid, fluid otherwise 
    """
    for i in range(n):
        for j in range(m):
            grid[i][j] = func(j,i)




def border_draw(grid,n,m):
    """
    Draws solid borders
    """
    for i in range(n):
        for j in range(m):
            if i == 0 or j == 0 or i == n-1 or j == m-1:
                grid[i][j] = True
