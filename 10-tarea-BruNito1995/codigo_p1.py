"""
En este archivo se resuelve la primera pregunta de la tarea 10 de metodos
numericos
Se pide encontrar la correlacion entre los PHD de matemática y la reserva de
uranio enriquecido

se responden las preguntas:
¿Cuanto deben bajar las reservas de Uranio para que nadie tenga un doctorado
en matematcias ?
¿Cuantos PhD se deberan graduar en un año para que las reservas de Uranio
bajaran a 0?

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sigma = 1  # asumimos que los errores distribuyen en forma iid

años = np.arange(1996, 2009)
phd = np.array([
    1122, 1123, 1177, 1083, 1050, 1010, 919, 993, 1076,
    1205, 1325, 1393, 1399])
U = np.array([
    66.1, 65.9, 65.8, 58.3, 54.8, 55.6, 53.5, 45.6, 57.7, 64.7, 77.5,
    81.2, 81.9])

# plot de ambos datos
plt.figure(1)
plt.clf()
plt.subplot(2, 1, 1)
plt.title("Ph.D.", size=15)
plt.xlabel("años", size=13)
plt.ylabel("Ph.D.", size=13)
plt.grid()
plt.plot(años, phd, '.')
plt.subplot(2, 1, 2)
plt.plot(años, U, '.k')
plt.title("Uranio [libras]", size=15)
plt.ylabel("Uranio [libras]", size=13)
plt.grid()
plt.xlabel("años", size=13)
plt.tight_layout()


plt.figure(2)
plt.clf()
plt.plot(U, phd, '.')
plt.title("Uranio y Ph.D.", size=15)
plt.xlabel("Uranio [libras]", size=13)
plt.ylabel("Ph.D.", size=13)
plt.grid()


# definimos la funcion del error de ajuste
def chi2(x_obs, y_obs, m, n):
    """
    Medida del error de ajuste. considera el error cuadratico
    """
    error = 0
    for i in range(len(x_obs)):
        error += (y_obs[i] - x_obs[i]*m+n)**2
    return error**0.5

# polinomio de orden un en funcion de U; P(U)
# phd = u*m + n
# en este caso x_obs corrseponde al Uranio e y_obs a los Ph.D
s = len(U)
sx = np.sum(U)
sy = np.sum(phd)
sxx = np.sum(U**2)
sxy = np.sum(U*phd)

m1 = (s*sxy - sx*sy)/(s*sxx - sx**2)
n1 = (sx*sxy - sxx*sy)/(sx**2 - s*sxx)


chi2_1 = chi2(U, phd, m1, n1)
baja_uranio = -n1/m1

p2b = """
Las reservas de Uranio enriquecido debiesen bajar a {} para que no hayan
doctorados en matematicas, segun la regresion con chi2 {}
""".format(baja_uranio, chi2_1)
print(p2b)

# polinomio de orden 1 en funcion de Phd; P(Phd)
# U = pHd*a + b
# en este caso x_obs corrseponde a los Ph.D. e y_obs al uranio

sx = np.sum(phd)
sy = np.sum(U)
sxx = np.sum(phd**2)

m2 = (s*sxy - sx*sy)/(s*sxx - sx**2)
n2 = (sx*sxy - sxx*sy)/(sx**2 - s*sxx)

chi2_2 = chi2(U, phd, m2, n2)
baja_phd = -n2/m2
p2c = """
Debiesen graduarse {} alumnos de Ph.D. para que las reservas de Uranio
enriquecido bajen a cero, segun la regresion lineal con chi2 {}
""".format(baja_phd, chi2_2)
print(p2c)

# verificar que los polinomios no son equivalentes
plt.figure(3)
plt.clf()
plt.plot(U, U*m1 + n1, label="modelo")
plt.plot(U, phd, '.', label="datos")
plt.legend()
plt.title("Ph.D. en funcion de U, regresion lineal", size=15)
plt.ylabel("Ph.D.", size=13)
plt.xlabel("Uranio [libras]", size=13)
plt.grid()


plt.figure(4)
plt.clf()
plt.plot(phd, phd*m2 + n2, label="modelo")
plt.plot(phd, U, '.', label="datos")
plt.title("U en funcion de Ph.D., regresion lineal", size=15)
plt.xlabel("Ph.D.", size=13)
plt.ylabel("Uranio [libras]", size=13)
plt.legend()
plt.grid()


# ambos modelos
plt.figure(5)
plt.clf()
plt.plot(U, U*m1 + n1, label="modelo 1")
plt.plot(phd*m2 + n2, phd, label="modelo 2")
plt.plot(U, phd, '.', label="datos")
plt.legend()
plt.title("Comparacion de modelos", size=15)
plt.xlabel("U", size=13)
plt.ylabel("Ph.D", size=13)
plt.grid()

# responder alreves las preguntas

p2d = """
Segun el primer modelo, seria necesario bajar las reservas de Uranio a {}
para que nadie obtenga doctorado en matematicas y segun el segundo modelo
serian necesarios {} Ph.D. para que bajen las reservas de uranio a 0
""".format(n2, n1)
print(p2d)


# polinamio evaluado en x de grado n
def pol(x, *args):
    output = 0
    for i in range(len(args)):
        output += x**i*args[i]

    return output

x = np.linspace(40, 80, 60)
y = np.linspace(900, 1500, 1000)

a_optimo, a_covarianza = curve_fit(
    f=pol, xdata=U, ydata=phd, p0=[0, 0, 0, 0, 0, 0], method='lm')
a1, a2, a3, a4, a5, a6 = a_optimo.copy()

b_optimo, b_covarianza = curve_fit(
    f=pol, xdata=phd, ydata=U, p0=[-10, -10, -10, -10, -10, -10], method='lm')
b1, b2, b3, b4, b5, b6 = b_optimo.copy()


plt.figure(6)
plt.clf()

plt.subplot(211)
plt.plot(x, pol(x, a1, a2, a3, a4, a5, a6), label="Modelo")
plt.plot(U, phd, '.', label="Datos")
plt.title("Ajuste de datos a un polinomio de orden 5", size=15)
plt.xlabel("Uranio [libras]", size=13)
plt.legend()
plt.ylabel("Ph.D.", size=13)
plt.grid()

plt.subplot(212)
plt.plot(y, pol(y, b1, b2, b3, b4, b5, b6), label="modelo")
plt.plot(phd, U, '.', label="Datos")
plt.ylabel("Uranio [libras]", size=13)
plt.xlabel("Ph.D.", size=13)
plt.legend()
plt.grid()
plt.tight_layout()


a_optimo, a_covarianza = curve_fit(
    f=pol, xdata=U, ydata=phd, p0=[0, 0, 0, 0, 0, 0, 0], method='lm')

a1, a2, a3, a4, a5, a6, a7 = a_optimo.copy()

b_optimo, b_covarianza = curve_fit(
    f=pol, xdata=phd, ydata=U, p0=[-100, -100, -100, -100, -100, -100, -100],
    method='lm')

b1, b2, b3, b4, b5, b6, b7 = b_optimo.copy()


plt.figure(7)
plt.clf()
plt.subplot(211)
plt.plot(x, pol(x, a1, a2, a3, a4, a5, a6, a7), label="Modelo")
plt.title("Ajuste de datos a un polinomio de orden 6", size=15)
plt.plot(U, phd, '.', label="Datos")
plt.xlabel("Uranio [libras]", size=13)
plt.ylabel("Ph.D.", size=13)
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(y, pol(y, b1, b2, b3, b4, b5, b6, b7), label="Modelo")
plt.plot(phd, U, '.', label="Datos")
plt.ylabel("Uranio [libras]", size=13)
plt.xlabel("Ph.D.", size=13)
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
