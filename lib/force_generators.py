import numpy as np

def linear_force(x,y,n,m, **params):
    intensity = float(params.get("intesity", 10))
    step = float(params.get("step", 0.1))
    return np.array([intensity - step*x, 0])

def half_linear_force(x,y,n,m, **params):
    intensity = float(params.get("intesity", 10))
    step = float(params.get("step", 0.25))
    end_x = int(params.get("end_x", 40))
    if x <= end_x:
        return np.array([intensity - step*x, 0])
    return np.array([0,0])

def constant_force(x,y,n,m, **params):
    intensity = float(params.get("intensity", 10))
    return np.array([intensity,0])

def half_constant_force(x,y,n,m, **params):
    intensity = float(params.get("intensity", 10))
    end_x = int(params.get("end_x", 40))
    if x <= end_x:
        return np.array([intensity,0])
    return np.array([0,0])

def narrow_stream_force(x,y,n,m, **params):
    intensity = float(params.get("intensity", 10))
    end_x = int(params.get("end_x", 40))
    start_y = int(params.get("start_y", 40))
    end_y = int(params.get("end_y", 60))
    if x <= end_x and y >= start_y and y <= end_y:
        return np.array([intensity, 0])
    return np.array([0,0])

def center_stream_force(x,y,n,m, **params):
    intensity = float(params.get("intensity", 2000))
    p = float(params.get("x", -1))
    q = float(params.get("y", 49.5))
    
    r2 = (x - p)**2 + (y-q)**2
    v = np.array([x-p,y-q]) / np.sqrt(r2)
    F = intensity/np.sqrt(r2)
    return F*v

