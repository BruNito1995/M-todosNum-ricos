"""
En este archivo se resuelve la pregunta 3 de la tarea 10 de metodos numericos.
Se intenta modelar la temperatura promedio planetaria y determinar en que año
se alcanza un aumento de 2 grados celcius
"""
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit import (minimize, Parameters, report_fit)


data = np.genfromtxt('GLB.txt')

# limpiamos Nan, considerando el promedio de los meses
# adyacentes
for i in range(139):
    for j in range(18):
        if np.isnan(data[i, j]):
            data[i, j] = data[i, j - 1] + data[i, j+1]

# manualmente ajustamos los nan que quedan en noviembre y diciembre
data[-1, 12] = data[-2, 12]
data[-1, 11] = data[-2, 11]
data[-1, 13] = data[-2, 13]

años = data[:, 0]
promedio = data[:, 13]
enero = data[:, 1]
febrero = data[:, 2]
marzo = data[:, 3]
abril = data[:, 4]
mayo = data[:, 5]
junio = data[:, 6]
julio = data[:, 7]
agosto = data[:, 8]
septiembre = data[:, 9]
octubre = data[:, 10]
noviembre = data[:, 11]
diciembre = data[:, 12]


plt.figure(1)
plt.clf()
plt.plot(años, promedio)
plt.xlabel("Años", size=13)
plt.ylabel("Temperatura", size=13)
plt.title("Temperatura promedio Anual", size=15)
plt.grid()
plt.show()


def parabola(x, a, b, c):
    output = a*x**2 + b*x + c
    return output


def sinusoide(x, A, f):
    output = A*np.sin(f*x)
    return output


def modelo(x, a, b, c, A, f):
    output = parabola(x, a, b, c) + sinusoide(x, A, f)
    return output


def residual(params, x, y):
    A = params['A'].value
    f = params['f'].value
    a = params['a'].value
    b = params['b'].value
    c = params['c'].value
    estimacion = modelo(x, a, b, c, A, f)
    return (y - estimacion)

# estimamos los parametros iniciales con suposiciones simples


params = Parameters()
params.add('A', value=1)
params.add('f', value=1)
params.add('a', value=1)
params.add('b', value=1)
params.add('c', value=1)

out = minimize(residual, params, args=(años, promedio))
report_fit(out)

A = float(out.params['A'])
f = float(out.params['f'])
a = float(out.params['a'])
b = float(out.params['b'])
c = float(out.params['c'])

extencion_años = np.arange(1880, 2080)
extencion_años_2 = np.arange(2018, 2080)
plt.figure(2)
plt.clf()
plt.axhline(y=2, label="2°C")
plt.plot(años, promedio, 'r', label="Datos")
plt.plot(extencion_años, modelo(extencion_años, a, b, c, A, f), 'g',
         label="Modelo")
plt.title("Prediccion de anomalia promedio anual de Temperatura", size=15)
plt.xlabel("Años", size=13)
plt.ylabel("Anomalia [°C]", size=13)
plt.grid()
plt.legend()
plt.plot()

ass = a + 6.9e-6
ai = a - 6.9e-6
bs = b + 0.026904
bi = b - 0.026904
cs = c + 26.21
ci = c - 26.21

plt.figure(3)
plt.clf()
plt.axhline(y=2, label="2°C")
plt.plot(años, promedio,  'r', label="Datos")
plt.plot(extencion_años, modelo(extencion_años, a, b, c, A, f),
         'g', label="Modelo Optimo")
plt.plot(extencion_años_2, 108.6 + modelo(extencion_años_2, ai, bi, ci, A, f),
         'k', label="Inervalo de Confianza")
plt.plot(extencion_años_2, -56.2 + modelo(extencion_años_2, ass, bs, ci, A, f),
         'k')
plt.title("Prediccion de anomalia promedio anual de Temperatura", size=15)
plt.xlabel("Años", size=13)
plt.ylabel("Anomalia [°C]", size=13)
plt.grid()
plt.legend()
plt.plot()


plt.show()
