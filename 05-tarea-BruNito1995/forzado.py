
# P2
# PENDULO FORZADO
"""
En este arichivo se resuelve la ecuacion del pendulo
forzado sin utilizar la aproximacion sin(alpha) = alpha
y a traves del metodo Runge-Kutta de orden 4
implementado en el archivo RK4.py
"""

import RK4
import numpy as np
import math
import matplotlib.pyplot as plt
m = 0.8*1.0271
L = 1.75*1.0271
Fo = 0.05*1.0271
g = 9.81
w0 = 1.726842105263158

# w0 = (g/L)**0.5/1.01
# defeinimos el tiempo
t = np.linspace(0, 80*math.pi, 10000)
h = t[1] - t[0]
y_rk4 = np.zeros((len(t), 2))
y_rk4[0] = 0.0


def pendulo_forzado(t, x):
    """
    recibe el tiempo y el vector x = [x,dx/dt]
    y calcula [w,dw/ds]
    Para el pendulo de van der pool
    El objetivo es usar esta impementacion Para
    resolver usando RK4
    """
    sin = math.sin
    cos = math.cos
    output = np.array((x[1], -g*sin(x[0])/L**2 + Fo*cos(w0*t)/m/L**2))
    return output

# resolvemos con runge kutta
for i in range(1, len(t)):
    y_rk4[i] = RK4.paso_rk4(pendulo_forzado, t[i-1], y_rk4[i-1], h)

print("maximo es " + str(np.max(y_rk4[:, 1])))
plt.close("all")
plt.plot(t, y_rk4[:, 1])
plt.xlabel("tiempo")
plt.ylabel("y(t)")
nombre = "Oscilador Forzado con $\omega$ = " + '{:6.3f}'.format(w0)
plt.title(nombre)
figname = "forzado" + '{:5.3f}'.format(w0) + '.jpg'
plt.savefig(figname)
plt.show()
