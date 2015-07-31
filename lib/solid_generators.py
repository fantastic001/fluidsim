

def circle_center(x,y):
    if (x - m/2)**2 + (y-n/2)**2 <= 8**2:
        return 1.0
    return 0.0

def square_center(x,y):
    if x >= 40 and x <= 60 and y >= 40 and y <= 60:
        return 1.0
    return 0.0

def blank(x,y):
    return 0.0