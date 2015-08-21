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
    return np.array([10,0])

def half_constant_velocity(x,y,n,m):
    if x <= 0.4*m:
        return np.array([10,0])
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
