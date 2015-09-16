
import scipy 
import scipy.sparse
import scipy.sparse.linalg
import numpy as np 


def K1(n, a11):
    """
    If a11 is 1 then matrix is suited for Neumann boundary conditions (pressure)
    If a11 is 2 then Dirchlet is used 
    If a11 is 3 then Dirchlet is used but for middle condition 
    """
    a = np.zeros(n-1)
    b = np.zeros(n)
    c = np.zeros(n-1)

    a.fill(-1)
    b[1:-1] = 2
    b[[0, -1]] = a11
    c.fill(-1)
    s = scipy.sparse.diags([a,b,c], [-1, 0, 1])
    return s

# Laplacian operators are computed as negative value of true laplacian operator 
# for pressure solution, then, it should be used like this: P = scipy.sparse.linalg.spsolve(-Lp, rhs)
# for diffusion, use u = scipy.sparse.linalg.spsolve(Lx, rhs)


def boundary_left(i,j, r, c, solids):
    return j == 0 or solids[i,j-1] 
def boundary_right(i,j, r, c, solids):
    return j == c-1 or solids[i,j+1] 
def boundary_up(i,j, r, c, solids):
    return i == 0 or solids[i-1,j] 
def boundary_down(i,j, r, c, solids):
    return i == r-1 or solids[i+1,j] 

def get_diffusion_solvers(viscosity, dt, n):
    SI = scipy.sparse.eye(n)
    SI_ = scipy.sparse.eye(n*(n-1))
    SI__= scipy.sparse.eye(n-1)
    
    Lxx = scipy.sparse.kron(SI,K1(n-1,2)) + scipy.sparse.kron(K1(n,3),SI__)
    Lx = SI_ + (viscosity*dt)*Lxx
    xsolver = scipy.sparse.linalg.splu(Lx)
    
    Lyy = scipy.sparse.kron(SI__,K1(n,3)) + scipy.sparse.kron(K1(n-1,2),SI)
    Ly = SI_ + (viscosity*dt)*Lyy
    ysolver = scipy.sparse.linalg.splu(Ly)
    return (xsolver, ysolver)

# ------ Grid manipulation functions ------

def to_centered(u,v):
    """
    u,v have to be transposed and with boundaries attached

    Use attach_boundaries(...) for attaching boundaries 
    """
    u,v = attach_boundaries(u,v)
    u[-1, :] = u[-2, :]
    u[0, :] = u[1, :]
    
    v[:, -1] = v[:,-2]
    v[:, 0] = v[:, 1]
    return ((u[:-1, :] + u[1:, :])/2, (v[:,:-1] + v[:,1:])/2)

def to_staggered(u,v):
    """
    Converts centered field to staggered field

    Returns staggered field WITHOUT boundaries
    """
    uc, vc = ((u[:-1, :] + u[1:, :])/2, (v[:,:-1] + v[:,1:])/2)
    #uc[[0, -1],:] = 2*u[[0, -1], :]
    #vc[:,[0, -1]] = 2*v[:, [0, -1]]
    return (uc, vc)

def field_transpose(u,v):
    return (u.T, v.T)


# ------ Following functions are suited for transposed grid (i.e. first index is x direction, second index is y direction) ------

def attach_boundaries(u,v):
    """
    Adds boundary nodes to x-component and y-component 

    u,v are transposed fields 
    """
    r,c = u.shape
    p = np.zeros([r+2,c])
    r,c = v.shape
    q = np.zeros([r, c+2])
    p[1:-1,:] = u
    q[:, 1:-1] = v
    return (p,q)

def compute_divergence(u,v):
    """
    Computes divergence 

    Fields must have boundaries attached by attach_boundaries(u,v)
    """
    return np.diff(u.T).T + np.diff(v)

def apply_pressure(u,v,p):
    """
    WARNING: Pressure must be negative such that it is computed from p = psolver.solve(rhs) not p = -psolver.solve(rhs)
    """
    return (u + np.diff(p.T).T, v + np.diff(p))

def projection(u,v, psolver, solids):
    n,m = solids.shape
    ubc, vbc = attach_boundaries(u,v)
    rhs = compute_divergence(ubc, vbc)[np.logical_not(solids)]
    rhs = np.array(rhs)
    count = np.count_nonzero(solids)
    count = n**2 - count
    rhs = rhs.reshape(count)
    p = np.zeros([n,n])
    res = psolver.solve(rhs)
    p[np.logical_not(solids)] = res
    #p = res.reshape([n,n])
    u,v = apply_pressure(u,v,p)
    return (u,v,p)


# ------ Soid handling ------

def reset_solids(u,v, solids):
    n,m = solids.shape
    ubc, vbc = attach_boundaries(u,v)
    for i in range(n):
        for j in range(n):
            if solids[i,j]:
                ubc[i+1,j] = 0
                ubc[i,j] = 0 
                vbc[i,j+1] = 0 
                vbc[i,j] = 0
    return (ubc[1:-1, :],vbc[:, 1:-1]) 

def set_solids(solids):
    n,m = solids.shape
    SI = scipy.sparse.eye(n)
    #Lp_ = scipy.sparse.kron(SI, K1(n, 1)) + scipy.sparse.kron(K1(n, 1), SI)
    Lp = scipy.sparse.lil_matrix((n**2,n**2))
    r,c = solids.shape
    count = np.count_nonzero(solids)
    Lp = scipy.sparse.lil_matrix((n**2,n**2))
    nonsolid_indices = []
    for i in range(r):
        for j in range(c):
            s = i*c + j
            if solids[i,j]:
                continue
            nonsolid_indices.append(s)
            Lp[s,s] = 0
            if not boundary_up(i,j, r, c, solids):
                Lp[s,s] += 1
                Lp[s,s-c] = -1
            if not boundary_down(i,j, r, c, solids):
                Lp[s,s] += 1
                Lp[s,s+c] = -1
            if not boundary_right(i,j, r, c, solids):
                Lp[s,s] += 1
                Lp[s,s+1] = -1
            if not boundary_left(i,j, r, c, solids):
                Lp[s,s] += 1
                Lp[s,s-1] = -1
    Lp[0,0] = 1.5 * Lp[0,0]
    Lp = Lp[np.ix_(nonsolid_indices, nonsolid_indices)]
    psolver = scipy.sparse.linalg.splu(Lp)
    return psolver

def boundary_count(i,j, n):
    s = 4
    if i == 0:
        s-=1
    if i == n-1:
        s-=1
    if j == 0:
        s-=1
    if j==n-1:
        s-=1
    return s

