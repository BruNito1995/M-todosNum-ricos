#!/usr/bin/env python
# -*- coding: utf-8 -*-

from planeta import Planeta, G, M, m
from RK4 import paso_rk4
import math
import numpy as np
from matplotlib import pyplot as plt
import pdb

condicion_inicial = [10,0,0,10]

p = Planeta(condicion_inicial)
a = p.energia_total()
print("energia = " + str(a))

def ecuacion_de_movimiento_rk4(t, Y):
    """
    Adaptacion de la función ecuacion_de_movimiento
    de la clase Planeta para ajustarse al modulo
    RK4 existe.

    Implementa la ecuación de movimiento, como sistema de ecuaciones de
    primer orden.
    El planeta se rige por la segunda ley de Newton en donde asumiremos
    que la unica fuerza que lo afecta es la atraccion gravitatoria
    del sol:

    F = ma = GmM/r^2

    Para obtener el sistema de ecuaciones se resuelve descomponiendo
    en los ejes cartesianos y resolviendo por separado

    Fx = mx'' = GmM/r^2*cos(theta)
    Fy = my'' = GmM/r^2*sin(theta)
    dx/dt = vx
    dy/dt = vy

    Donde r es la distancia al sol y theta el angulo
    que barre el vector posicion

    Y = [x, y, vx, vy]
    retorna [vx, vy, x'',y'']
    """
    x, y, vx, vy = Y
    #pdb.set_trace()
    theta = math.atan2(y,x)
    r = (x**2 + y**2)**0.5
    fx = -G*M*math.cos(theta)*r**(-2)
    fy = -G*M*math.sin(theta)*r**(-2)

    output = np.array([vx, vy, fx, fy])

    return output

t = np.linspace(0,100,10000)
H = t[1] - t[0]
w = np.zeros((len(t),4))
w[0] = condicion_inicial

for i in range(len(t)-1):
    w[i+1] = paso_rk4(ecuacion_de_movimiento_rk4, t[i], w[i], H)
    p.y_actual = w[i+1]
    p.t_actual = t[i]

plt.close(1)
plt.figure(1)
plt.plot(w[:,0], w[:,1], label = "Trayectoria")
plt.plot(w[:,2], w[:,3], label = "Velocidad")
plt.title("Trayectoria y velocidad del planeta")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.plot(0,0,'ro')
plt.grid()
plt.savefig("trayectoria_rk4.png")


plt.close(2)
r = (w[:,0]**2 + w[:,1]**2)**0.5
plt.figure(2)
plt.plot(t,r, label="Distancia sol")
plt.title("Distancia en funcion del tiempo")
plt.legend()
plt.xlabel("tiempo")
plt.ylabel("Distancia")
plt.grid()
plt.show()
