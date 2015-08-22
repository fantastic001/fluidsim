import numpy as np

def linear_velocity(x,y,n,m):
    return np.array([10 - 0.1*x, 0])

def half_linear_velocity(x,y,n,m):
    if x <= 0.40*m:
        return np.array([10 - 0.25*x, 0])
    return np.array([0,0])

def linear_velocity_small(x,y,n,m):
    return np.array([0.1 - 0.001*x, 0])

def half_linear_velocity_small(x,y,n,m):
    if x <= 0.40*m:
        return np.array([0.1 - 0.001*2.5*x, 0])
    return np.array([0,0])

def constant_velocity(x,y,n,m):
    return np.array([1,0])

def half_constant_velocity(x,y,n,m):
    if x <= 0.4*m:
        return np.array([1,0])
    return np.array([0,0])

def constant_velocity_small(x,y,n,m):
    return np.array([0.1,0])

def half_constant_velocity_small(x,y,n,m):
    if x <= 0.4*m:
        return np.array([0.1,0])
    return np.array([0,0])

def narrow_stream_velocity(x,y,n,m):
    if x <= 40 and y >= 40 and y <= 60:
        return np.array([10, 0])
    return np.array([0,0])

def center_stream_velocity(x,y,n,m):
    r2 = (x + 1)**2 + (y-0.495*n)**2
    v = np.array([x+1,y-0.495*n]) / np.sqrt(r2)
    F = 20/np.sqrt(r2)
    return F*v

def down_stream_velocity(x,y,n,m):
    r2 = (x + 1)**2 + (y-n)**2
    v = np.array([x+1,y-n]) / np.sqrt(r2)
    F = 20/np.sqrt(r2)
    return F*v
