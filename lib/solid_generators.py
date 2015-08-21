

def circle_center(x,y,n,m):
    if (x - 0.495*m)**2 + (y-0.495*n)**2 <= 5**2:
        return 1.0
    return 0.0

def square_center(x,y,n,m):
    if x >= 40 and x <= 60 and y >= 40 and y <= 60:
        return 1.0
    return 0.0

def blank(x,y,n,m):
    return 0.0

def slide(x,y,n,m):
    if 45 <= x and x <= 54 and y <= 45 and y >= 54:
        return 1.0
    return 0.0
