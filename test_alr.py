import numpy as np
from pyamg.gallery import poisson
from alr import compute_alr

nx = 32
a = -poisson((nx, nx))

t = np.linspace(0, 1, nx+1)[1:nx+1]
x, y = np.meshgrid(t, t)
f = np.exp((-1.0 * (x - 0.5) ** 2 - 1.5 * (y - 0.7) ** 2))
y0 = f.flatten().reshape((-1, 1))
if y0.T.dot(a.dot(y0)) > 0:
    a = -a
a = a.tocsr()

result = compute_alr(a, y0)
print 'Reached accurracy is ', result['resnorm']
