""""
En este archivo se resuelve el problema de la tarea 11. Se busca modelar el
registro del archvivo espectro.data con 2 modelos distintos:

modelo 1: Gaussiana + recta => 5 parametros
modelo 2: Lorentz + recta => 5 parametros

luego, se compara la bodad de ajuste con chi^2 (medida de error de ajuste) y
Kolmogorov Smirnov (probabilidad de que se cumpla la hipotesis nula, asumiendo
un comportamiento aproximadamente gaussiano de los errores)
"""


import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit import (minimize, Parameters, report_fit)
import scipy.stats


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


def lorenz(x, amplitud, promedio, sigma):
    output = amplitud*scipy.stats.cauchy(loc=promedio, scale=sigma).pdf(x)
    return output


def modelo1(x, A, mu, sigma, m, n):
    """
    Retorna el valor de la funcion correspondiente al primer modelo (gauss +
    recta)
    """
    return gauss(x, A, mu, sigma) + recta(x, m, n)


def modelo2(x, A, mu, sigma, m, n):
    """
    Retorna el valor de la funcion correspondiente al segundo modelo (lorenz +
    recta)
    """
    return lorenz(x, A, mu, sigma) + recta(x, m, n)


def residual1(params, x, y):
    """
    calcula la residual del modelo1
    """
    m = params['m'].value
    n = params['n'].value
    sigma = params['s'].value
    A = params['A'].value
    mu = params['mu'].value
    estimacion = modelo1(x, A, mu, sigma, m, n)
    return (y - estimacion)


def residual2(params, x, y):
    """
    Calcula la residual del modelo2
    """
    m = params['m'].value
    n = params['n'].value
    sigma = params['s'].value
    A = params['A'].value
    mu = params['mu'].value
    estimacion = modelo2(x, A, mu, sigma, m, n)
    return (y - estimacion)

# estimamos los parametros iniciales con suposiciones simples

m0 = (radiacion[0] - radiacion[-1]) / (longitud[0] - longitud[-1])
n0 = (radiacion[0]) - m0*longitud[0]
s0 = 4  # por inspeccion
A0 = np.max(radiacion) - np.min(radiacion)
mu0 = longitud[np.argmax(radiacion)]

params1 = Parameters()
params1.add('m', value=m0)
params1.add('n', value=n0)
params1.add('s', value=s0)
params1.add('A', value=A0)
params1.add('mu', value=mu0)

amplificacion = 1e5
params2 = Parameters()
params2.add('m', value=m0)
params2.add('n', value=n0)
params2.add('s', value=s0)
params2.add('A', value=A0)
params2.add('mu', value=mu0)

out1 = minimize(residual1, params1, args=(longitud, radiacion))
report_fit(out1)

out2 = minimize(residual2, params2, args=(longitud, radiacion))
report_fit(out2)

A1 = float(out1.params['A'])
mu1 = float(out1.params['mu'])
s1 = float(out1.params['s'])
m1 = float(out1.params['m'])
n1 = float(out1.params['n'])

A2 = float(out2.params['A'])
mu2 = float(out2.params['mu'])
s2 = float(out2.params['s'])
m2 = float(out2.params['m'])
n2 = float(out2.params['n'])

# figura2
plt.figure(2)
plt.clf()
plt.title("Ajuste de curva de Radiacion, MODELO 1", size=15)
plt.xlabel("Longitud de onda [Armstrong]", size=13)
plt.ylabel(r"$F_{\nu} [erg s^{-1} Hz^{-1} cm^{-2}$]", size=13)
plt.plot(longitud, radiacion, '.', label="Datos")
plt.plot(
    longitud, modelo1(longitud, A0, mu0, s0, m0, n0), label="Modelo inical")
plt.plot(
    longitud, modelo1(longitud, A1, mu1, s1, m1, n1), label="Modelo final")
plt.grid()
plt.legend()
plt.plot()

# figura3
plt.figure(3)
plt.clf()
plt.title("Ajuste de curva de Radiacion, MODELO 2", size=15)
plt.xlabel("Longitud de onda [Armstrong]", size=13)
plt.ylabel(r"$F_{\nu} [erg s^{-1} Hz^{-1} cm^{-2}$]", size=13)
plt.plot(longitud, radiacion, '.', label="Datos")
plt.plot(
    longitud, modelo2(longitud, A0, mu0, s0, m0, n0), label="Modelo inical")
plt.plot(
    longitud, modelo2(longitud, A2, mu2, s2, m2, n2), label="Modelo final")
plt.grid()
plt.legend()
plt.plot()

# figura4
plt.figure(4)
plt.clf()
plt.title("Comparacion de Ajuste de modelos de radiacion", size=15)
plt.xlabel("Longitud de onda [Armstrong]", size=13)
plt.ylabel(r"$F_{\nu} [erg s^{-1} Hz^{-1} cm^{-2}$]", size=13)
plt.plot(longitud, radiacion, '.', label="Datos")
plt.plot(
    longitud, modelo1(
        longitud, A1, mu1, s1, m1, n1), 'r', label="Modelo 1 (Gaussiana)")
plt.plot(longitud, modelo2(
    longitud, A2, mu2, s2, m2, n2), 'k',  label="Modelo 2 (Lorenz)")
plt.grid()
plt.legend()
chi2_red1 = np.sum(out1.residual**2)/5
chi2_red2 = np.sum(out2.residual**2)/5

reporte = """
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Para el MODELO 1, Gaussiano:
chi^2 reducido de = {:.2e}
con parametros:
A = {:.2e}
mu = {:.2e}
sigma = {:.2e}
m = {:.2e}
n = {:.2e}
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Para el MODELO 2, Lorenz:
chi^2 reducido = {:.2e}
A = {:.2e}
mu = {:.2e}
sigma = {:.2e}
m = {:.2e}
n = {:.2e}
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
""".format(chi2_red1, A1, mu1, s1, m1, n1, chi2_red2, A2, mu2, s2, m2, n2)

print(reporte)


# Kolmogorov Smirnov
xmin = np.min(longitud)
xmax = np.max(longitud)
sort_m1 = np.sort(modelo1(np.linspace(xmin, xmax, 1000), A1, mu1, s1, m1, n1))
sort_m2 = np.sort(modelo2(np.linspace(xmin, xmax, 1000), A2, mu2, s2, m2, n2))
sort_data = np.sort(radiacion)

CDF_model1 = np.array([np.sum(sort_m1 <= yy) for yy in sort_data])/len(sort_m1)
CDF_model2 = np.array([np.sum(sort_m2 <= yy) for yy in sort_data])/len(sort_m2)

N = len(sort_data)

plt.figure()
plt.clf()
plt.plot(sort_data, np.arange(N) / N, '-^', drawstyle='steps-post')
plt.plot(sort_data, np.arange(1, N+1) / N, '-.', drawstyle='steps-post')
plt.plot(sort_data, CDF_model1, '-x', drawstyle='steps-post')
plt.grid()

plt.figure()
plt.clf()
plt.plot(sort_data, np.arange(N) / N, '-^', drawstyle='steps-post')
plt.plot(sort_data, np.arange(1, N+1) / N, '-.', drawstyle='steps-post')
plt.plot(sort_data, CDF_model2, '-x', drawstyle='steps-post')
plt.grid()

max1_1 = np.max(CDF_model1 - np.arange(N) / N)
max1_2 = np.max(np.arange(1, N+1)/N - CDF_model1)
max2_1 = np.max(CDF_model2 - np.arange(N) / N)
max2_2 = np.max(np.arange(1, N+1)/N - CDF_model2)

Dn1 = max(max1_1, max1_2)
Dn2 = max(max2_1, max2_2)
print("Dn para nuestro modelo1 = ", Dn1)
print("Dn para modelo 2 = ", Dn2)


ks_dist = kstwobign()
alpha = 0.05
Dn_critico = ks_dist.ppf(1 - alpha) / np.sqrt(len(radiacion))
print("Dn critico = ", Dn_critico)

print("Nivel de confianza, modelo 1 : ", 1 - ks_dist.cdf(Dn1 * np.sqrt(N)))
print("Nivel de confianza, modelo 2 : ", 1 - ks_dist.cdf(Dn2 * np.sqrt(N)))
plt.show()
