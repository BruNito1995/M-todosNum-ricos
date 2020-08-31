"""
En este documento se resuelve la tarea 7 de metodos numericos la cual consiste
en modelar el comportamiento de una especie animal, a traves de la ecuacion
Fisher - KPP (reaccion - difusion) en su version unidimensional

dn/dt = Y d2n/dt2 + un - un^2

donde n(t,x) es la densidad de la especie.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# constantes del problema
Y = 0.001
u = 1.5

Lx = 1
Nx = 500
hx = Lx/(Nx-1)

ht = hx**2 / 2 / Y
ht = ht*0.9
# ht = ht*1000  # inestable
Lt = 8
Nt = int(Lt/ht)

# condicion inicial
x = np.linspace(0, Lx, Nx)
n_0 = np.exp(-x**2/0.1)


# CONDICION INICIAL
plt.figure(1)
plt.clf()
plt.plot(x, n_0)
plt.title("Condicion inicial de la poblacion", size=15)
plt.xlabel("Distancia", size=13)
plt.ylabel("Densidad de Poblacion", size=13)
plt.grid()

# iniciamos n y las variables de CN
b = np.zeros(Nx)
alpha = np.zeros(Nx)
beta = np.zeros(Nx)
n = n_0.copy()
n_archive = np.zeros((Nt, Nx))
n_archive[0] = n.copy()
s = ht*Y / hx**2 / 2


# DEFINIMOS CRANCK NICHOLSON
def llena_b(b, T, Nx, s):
    for i in range(1, Nx-1):
        b[i] = s * n[i+1] + (1 - 2*s)*n[i] + s*n[i-1]


def llena_alpha_beta(alpha, beta, b, Nx, s):
    alpha[0] = 0
    beta[0] = 1  # Ver si va 0 o 1 aqui
    for i in range(1, Nx):
        alpha[i] = s / (-s * alpha[i-1] + (2*s + 1))
        beta[i] = (b[i] + s * beta[i-1]) / (-s * alpha[i-1] + (2 * s + 1))


def avanza_CN(alpha, beta, Nx):
    """
    Utilizando el metodo de Cranck - Nicholson
    da un paso temporal para el arreglo n
    de la forma:
    n[i] = alpha*n[i+1] + beta

    n: vector de densidad de poblacion (argumento implicito)
    alpha y beta: constantes de ajuste para el paso de CN
    Nx: numero de pasos discretos que tiene n
    """
    # imponemos condicion de borde
    n[-1] = 0
    n[0] = 1
    for i in range(Nx-2, 0, -1):
        n[i] = alpha[i]*n[i+1] + beta[i]


# EULER EXPLICITO
def avanza_euler(n, Nx):
    """
    Da un paso temporal para los terminos reactivos de la ecuacion que regula
    la densidad de poblacion, a traves del metodo explicito de Euler,
    aproximando
    dn/dt = n[i+1] - n[i] / h
    n: vecotr de densidad de Poblacion
    Nx: numero de discretizaciones longitudinales
    la funcion actualiza n y no retorna nada
    """
    for i in range(1, Nx-1):
        n[i] = n[i]*(u*ht + 1) - n[i]**2*u*ht

# ITERACIONES DEL MODELO
for j in range(1, Nt):
    llena_b(b, n, Nx, s)
    llena_alpha_beta(alpha, beta, b, Nx, s)
    avanza_CN(alpha, beta, Nx)
    avanza_euler(n, Nx)
    n_archive[j] = n.copy()

# EVOLUCION TEMPORAL
plt.figure(2)
plt.clf()
plt.imshow(n_archive, origin="lower", aspect="auto", extent=[0, Nx, 0, Lt])
plt.colorbar(label="Densidad de Poblacion")
plt.title("Evolucion Temporal de la Densidad de \n Poblacion, paso inestable"
          , size=15)
plt.xlabel("Distancia", size=13)
plt.ylabel("Tiempo", size=13)

# CONDICION FINAL
plt.figure(3)
plt.clf()
plt.title("Poblacion Final", size=15)
plt.xlabel("Distancia", size=13)
plt.ylabel("Densidad de Poblacion", size=13)
plt.plot(x, n, 'r')
plt.grid()

# VARIACION (FINAL - INICIAL)
plt.figure(4)
plt.clf()
plt.title("Variacion Poblacion", size=15)
plt.xlabel("Distancia", size=13)
plt.ylabel("Densidad de Poblacion", size=13)
plt.grid()
plt.fill(x, n-n_0, 'g', label="diferencia")
plt.plot(x, n, label="final")
plt.plot(x, n_0, label="inicial")
plt.legend()

plt.show()
