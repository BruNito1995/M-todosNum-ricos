"""
En este archivo se resuelve la pregunta 2 de la tarea 9 de metodos numericos.
Se solicita encontrar la densidad de galaxias con masa M = 10**11Mo utilizando
la funcion de Schechter

P(L)dL = P*(L/L*)^a*exp(-L/L*)dL

donde P(L) es la densidad de galaxias con luminosidad entre L y L + dL
"""
import numpy as np
import matplotlib.pyplot as plt

Lsol = 3.82e26 # en watt
Msol =  1.98e30 # en kg

Pc = 10e-3 # en megaparsecs cubico
Lc = 10e10
a = -1

# consideramos FL como fdp
def lum(L):
    exp = np.exp
    output = Pc*(L/Lc)**a*exp(-L/Lc)
    return output

x_to_plot = np.logspace(9,12, base=10)

plt.figure(1)
plt.clf()
plt.loglog(x_to_plot, lum(x_to_plot))
plt.title("Funciion de Luminosidad")
plt.xlabel(r'$\Phi$')
plt.show()
