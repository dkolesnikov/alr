import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

def compute_alr(a, y0, max_iter = 50, acc=1e-8):

    i = 0
    id_n = sp.eye(a.shape[0])

    u = y0 / la.norm(y0)
    b = u.T.dot(a.dot(u))
    toler = la.norm(a.dot(u)/b - u) * acc 
    
    u = np.linalg.qr(u, mode='reduced')[0]
    w = a.dot(u) - u.dot(b)
    w /= la.norm(w)
    w = w.reshape(-1, 1)

    while True:
        au = a.dot(u)
        b = u.T.dot(au)
        yr = u.T.dot(y0)
        z0 = la.solve_lyapunov(b, yr.dot(yr.T))
        w1 = a.dot(w)
        w1 = w1 - u.dot(u.T.dot(w1))
        q = la.norm(w1) * la.norm(z0[0, :])
        if q < toler:
            break
        if i >= max_iter:
            break
            
        qn = z0[0, :].reshape(1, -1)
        qn /= la.norm(qn)
        
        bn = qn.dot(b).dot(qn.T)
        v = sla.spsolve(a + bn[0, 0] * id_n, w).reshape(-1, 1)
        
        v -= u.dot(u.T.dot(v))
        v /= la.norm(v)
        v -= u.dot(u.T.dot(v))
        v /= la.norm(v)
        u = np.hstack((v, u))
        
        w1 = w.copy()
        w1 -= u.dot(u.T.dot(w1))
        w1 /= la.norm(w1)
        w1 -= u.dot(u.T.dot(w1))
        w1 /= la.norm(w1)
        u = np.hstack((u, w1))
        
        w = a.dot(w)
        w -= u.dot(u.T.dot(w))
        w_norm = la.norm(w)
        w /= la.norm(w)
        w -= u.dot(u.T.dot(w))
        w /= la.norm(w)
        
        i += 1
    return {'resnorm': q, 'u': u}
