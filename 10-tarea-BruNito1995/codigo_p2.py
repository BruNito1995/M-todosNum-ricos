"""
En este archivo se resuelve la pregunta 2 de la tarea 10 de metodos numericos.
Se intenta modelar la linea de absorsión sel archivo espectro.dat como una
suma de una funcion lineal con una campana de Gauss
"""
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit import (minimize, Parameters, report_fit)

data = np.genfromtxt('espectro.dat')
radiacion = data[:, 1]
longitud = data[:, 0]

plt.figure(1)
plt.clf()
plt.plot(longitud, radiacion)
plt.title("Espectro de radiación", size=15)
plt.xlabel("Longitud de onda [Armstrong]", size=13)
plt.ylabel(r"$F_{\nu} [erg s^{-1} Hz^{-1} cm^{-2}$]", size=13)
plt.grid()


def gauss(x, amplitud, promedio, sigma):

    exponente = -(x - promedio)**2/2/sigma**2
    output = amplitud*np.exp(exponente)
    return output


def recta(x, pendiente, coef):
    return x*pendiente + coef


def modelo(x, A, mu, sigma, m, n):
    return gauss(x, A, mu, sigma) + recta(x, m, n)


def residual(params, x, y):
    m = params['m'].value
    n = params['n'].value
    sigma = params['s'].value
    A = params['A'].value
    mu = params['mu'].value
    estimacion = modelo(x, A, mu, sigma, m, n)
    return (y - estimacion)

# estimamos los parametros iniciales con suposiciones simples

m0 = (radiacion[0] - radiacion[-1]) / (longitud[0] - longitud[-1])
n0 = (radiacion[0]) - m0*longitud[0]
s0 = 4  # por inspeccion
A0 = np.max(radiacion) - np.min(radiacion)
mu0 = longitud[np.argmax(radiacion)]

params = Parameters()
params.add('m', value=m0)
params.add('n', value=n0)
params.add('s', value=s0)
params.add('A', value=A0)
params.add('mu', value=mu0)

out = minimize(residual, params, args=(longitud, radiacion))
report_fit(out)

Af = float(out.params['A'])
muf = float(out.params['mu'])
sf = float(out.params['s'])
mf = float(out.params['m'])
nf = float(out.params['n'])

plt.figure(2)
plt.clf()
plt.title("Ajuste de curva de Radiacion", size=15)
plt.xlabel("Longitud de onda [Armstrong]", size=13)
plt.ylabel(r"$F_{\nu} [erg s^{-1} Hz^{-1} cm^{-2}$]", size=13)
plt.plot(longitud, radiacion, '.', label="Datos")
plt.plot(
    longitud, modelo(longitud, A0, mu0, s0, m0, n0), label="Modelo inical")
plt.plot(longitud, modelo(longitud, Af, muf, sf, mf, nf), label="Modelo final")
plt.grid()
plt.legend()
plt.plot()

chi2_red = np.sum(out.residual**2)/5

plt.show()
