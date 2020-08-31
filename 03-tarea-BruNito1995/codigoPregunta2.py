import numpy as np
import scipy.interpolate as spl
import matplotlib.pyplot as plt

datos  = np.loadtxt('textoDatos.txt')

años = []
#generamos el vector de años
for x in range(2008,2018):
    if x != 2011:
        años.append(x)

anomalias = []
#generamos el vector de anomalías
for x in range(len(años)):
    anomalias.append(datos[x][13])

spline = spl.InterpolatedUnivariateSpline(años,anomalias, k = 5)
print('La interpolación de la anomalía del 2011 es: ' +
  str(spline(2011)))
print('El error es de ' +
  str(spline(2011) - 0.58))

print('La interpolación de la anomalía del 2018 es: ' +
  str(spline(2018)))

x = np.linspace(2008,2018,100)
ajuste_spline = []
for i in range(len(x)):
    ajuste_spline.append(spline(x[i]))

plt.figure()
plt.plot(x,ajuste_spline)
plt.plot(años,anomalias,'ro')
plt.title("Anomalía Climática")
plt.xlabel("tiempo[años]");plt.ylabel('Anomalía[°C]')
plt.savefig("interpolacion_anomalia.png")
plt.show()
