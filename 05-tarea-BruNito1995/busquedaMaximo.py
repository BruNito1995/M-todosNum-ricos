import numpy as np
import forzado
import math
import RK4

iteraciones = 20
m = 0.8*1.0271
L = 1.75*1.0271
Fo = 0.05*1.0271
g = 9.81
t = np.linspace(0,40*math.pi,10000)
h = t[1] - t[0]
y_rk4 = np.zeros((len(t),2))
y_rk4[0] = 0.0
w0 = (g/L)**0.5
w1 = 0
#w = np.linspace(w0,w1,iteraciones)

w = np.linspace(1.72,1.73,iteraciones)
i = 0
amp_max = []
while i < iteraciones:
    w0 = w[i]

    def pendulo_forzado(t,x):
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

    #runge Kutta
    for j in range(1,len(t)):
        y_rk4[j] = RK4.paso_rk4(pendulo_forzado, t[j-1], y_rk4[j-1],h)

    #maximo
    amp_max.append(np.max(y_rk4[:,1]))
    i += 1


amplitud = np.max(amp_max)
indice = np.argmax(amp_max)
frecuencia = w[indice]
print('Amplitud maxima = ' + str(amplitud))
print('frecuencia = ' +  str(frecuencia) + ' indice = ' + str(indice))
ww = np.linspace(w[indice-1],w[indice+1],iteraciones)
