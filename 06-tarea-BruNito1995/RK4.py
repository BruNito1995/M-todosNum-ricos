"""
En este archivo se implementa el
metodo de Runge-Kutta de orden 4,
para resolver ecuaciones diferenciales
de primer orden
Este metodo se puede adaptar facilmente
para resolver ecuaciones de mayor orden
a traves de cambios de variable que
permitan describir el problema como
un sistema de ecuaciones de primer orden
"""

# importamos modulos necesarios
import numpy as np


def calcula_k1(func, x, y):
    k1 = func(x, y)
    return k1


def calcula_k2(func, x, y, h):
    k1 = calcula_k1(func, x, y)
    k2 = func(x + h/2, y + k1*h/2)
    return k2


def calcula_k3(func, x, y, h):
    k2 = calcula_k2(func, x, y, h)
    k3 = func(x + h/2, y + k2*h/2)
    return k3


def calcula_k4(func, x, y, h):
    k3 = calcula_k3(func, x, y, h)
    k4 = func(x + h, y + k3*h)
    return k4


def paso_rk4(func, x, y, h):
    """
    Esta funcion calcula los coeficientes de RK4 para dar un paso
    en la discrecion. Es necesario otorgar condiciones iniciales
    al problema para resolver.

    si anotamos:
    dy/dt = func(t,y)

    entonces:
    y[n+1] = y[n] + (k1 + 2k2 + 3k3 + k4)/6

    En el caso que se quiera resolver para una EDO
    de orden mayor, es necesario definir y en forma vectorial

    EJEMPLO: orden 2
    d2x/dt2 = x*cos(t)
    definimos dx/dt = w
    y el sistema queda expresado
    y = d/dt(x , w) = (w , x*cos(t))
    """
    # calculamos coeficientes
    k1 = calcula_k1(func, x, y)
    k2 = calcula_k2(func, x, y, h)
    k3 = calcula_k3(func, x, y, h)
    k4 = calcula_k4(func, x, y, h)
    # realizamos el paso
    y_nuevo = y + h*(k1 + 2*k2 + 2*k3 + k4)/6

    return y_nuevo
