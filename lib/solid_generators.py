
import numpy as np 

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

def wing(x,y,n,m):
    r = 0.4*n
    w = 0.40*n
    h = 0.1*m
    xa = r*np.cos(np.pi / 4)
    ya = xa = r*np.sin(np.pi / 4)
    ya = 0.60*n
    theta = 25 * (np.pi / 180)
    xb = xa + h*np.cos(theta)
    yb = ya + h*np.sin(theta)
    xc = xa + w*np.sin(theta)
    yc = ya - w*np.cos(theta)
    ab = np.array([xb - xa, yb - ya])
    ac = np.array([xc - xa, yc - ya])
    v = np.array([x - xa, y - ya])
    if v.dot(ab) >= 0 and v.dot(ab) <= h**2 and v.dot(ac) >= 0 and v.dot(ac) <= w**2:
        return 1
    else:
        return 0 
