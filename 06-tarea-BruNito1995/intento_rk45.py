from scipy.integrate import RK45
from solucion_usando_rk4 import ecuacion_de_movimiento_rk4
import numpy as np
from matplotlib import pyplot as plt
condicion_inicial = [10,0,0,10]

w = RK45(ecuacion_de_movimiento_rk4,0,condicion_inicial,1e6)
Nmax = 2000
i = 0
j = np.zeros((Nmax,4))
while i < Nmax:
    j[i,:] = w.y
    w.step()
    i += 1

x = j[:,0]
y = j[:,1]

plt.close(1)
plt.figure(1)
plt.plot(x,y)
plt.plot(0,0,'ro')
plt.show()
