"""
En este codigo se resuelve el problema planteado
en la tarea 7 de metodos numericos
que consiste en utilizar el metodo de la relajacion
para resolver el problema de la temperatura atmosferica
debido a una planteado
"""

# primero definiremos las temperaturas de
# cada lugar
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
w = 1.5


def heaviside(t):
    if t > 0:
        return 1
    elif t == 0:
        return 0.5
    else:
        return 0


def temperatura_superficie_mar(t):
    """
    Calcula la temperatura de la superficie del mar
    en funcion del tiempo de acuero a:

    T = 4 entre las 0 y las 8
      = 4 + 16*(t-8)/(16-8) entre las 8 y las 16
      = 20 - 4*(t-16)/(16-24) entre las 16 y las 24
    """
    H = heaviside
    t = t % 24
    c1 = 16*(t-8)/(16-8)
    c2 = 16 - 16*(t - 16)/(24 - 16)
    temperatura = 4 + c1*H(t-8)*H(16-t) + c2*H(t-16)

    return temperatura


def temperatura_chimenea(t):
    """
    Calcula la temperatura a la que emite la chimenea
    de acuerdo a T = 500(cos(pi/12*t) + 2)
    """
    # ajustamos al dia
    t = t % 24
    pi = math.pi
    cos = math.cos
    factor_tiempo = cos(pi*t/12)
    temperatura = 500*(factor_tiempo + 2)

    return temperatura


def temperatura_suelo(z):
    """
    Calcula la temperatura que tiene el suelo, en función de su altura
    considerando que sobre los 1800 msnm la temperatura es 0 C
    y bajo los 1800 msnm la temperatura es 10 C
    """
    if z < 1800:
        return 15
    else:
        return 0


def temperatura_atmosfera(z, t):
    """
    Calcula la temperatura de la
    atmosfera en funcion del tiempo y la altura
    """
    t_mar = temperatura_superficie_mar(t)
    temperatura = t_mar - 6*z/1000
    return temperatura


t_plot = np.linspace(0, 48, 1000)
x_plot = np.linspace(0, 4000, 8000)

# GEOGRAFIA

# LONGITUDES
sup_mar = 1200 + 400*0.271
costa = 300
subida1 = 800
bajada1 = 300
subida2 = 500
bajada2 = 4000 - sup_mar - costa - subida1 - bajada1 - subida2
chimenea = 100

# ALTURAS
cima_costa = 100
cima1 = 1500 + 200*0.271
cima2 = 1850 + 100*0.271
valle1 = 1300 + 200*0.271
valle2 = cima1 - 100


def geografia(x):
    """
    Esta función estima la altura dela geografia del esquema en funcion de la
    longitud horizontal
    """

    # superficie mar
    if x < sup_mar:
        altura = 0
        return altura

    # Chimenea
    if x < sup_mar + chimenea:
        altura = 0
        return altura

    # costa, inclinacion
    elif x < sup_mar + chimenea + costa:
        altura = (x - sup_mar - chimenea)/3
        return altura

    # cordillera de la costa, primera cima
    elif x < sup_mar + chimenea + costa + subida1:
        altura = cima_costa + (cima1-cima_costa)*(
            x - sup_mar - chimenea - costa)/subida1
        return altura

    # bajada al valle
    elif x < sup_mar + costa + subida1 + bajada1 + chimenea:
        altura = cima1 - (cima1-valle1)*(
            x - sup_mar - chimenea - costa - subida1)/bajada1
        return altura

    # subiendo cordillera andes
    elif x < sup_mar + costa + subida1 + bajada1 + subida2 + chimenea:
        altura = valle1 + (cima2-valle1)*(
            x - sup_mar - chimenea - costa - subida1 - bajada1)/subida2
        return altura

    # bajando cordillera andes
    elif x < (
     sup_mar + costa + subida1 + bajada1 + subida2 + bajada2 + chimenea):
        altura = cima2 - (cima2 - valle2)*(
             x - sup_mar - chimenea - costa - subida1 - bajada1 - subida2
                 )/bajada2
        return altura


def CDB(x, t):
    """
    define el tipo de condicion de borde que aplica
    en la zona segun la posicion horizontal y
    calcula la temperatura segun el tiempo
    """
    if x < sup_mar:
        # estamos sobre el mar
        return temperatura_superficie_mar(t)

    elif x < sup_mar + chimenea:
        # estamos en la Chimenea
        return temperatura_chimenea(t)

    else:
        # estamos en la tierra
        z = geografia(x)
        temperatura = temperatura_suelo(z)
        return temperatura


# iniciamos las variables
T_mar = []
T_chimenea = []
altura_geografia = []
T_atmosfera = []

# iniciamos las temperaturas de las condiciones de borde para graficar
# de mar chimenea y el prefil geografico
for tiempo in t_plot:
    T_mar.append(temperatura_superficie_mar(tiempo))

for tiempo in t_plot:
    T_chimenea.append(temperatura_chimenea(tiempo))

for longitud in x_plot:
    altura_geografia.append(geografia(longitud))

for altura in altura_geografia:
    T_atmosfera.append(temperatura_atmosfera(altura, 0))

# FIGURAS
plt.figure(1)
plt.clf()
plt.subplot(311)
plt.plot(t_plot, T_mar, 'b')
plt.grid()
plt.xlabel("tiempo [horas]")
plt.ylabel("Temperatura [°C]")
plt.title("Temperatura superficie del mar")

plt.subplot(312)
plt.plot(t_plot, T_chimenea, 'r')
plt.grid()
plt.xlabel("tiempo [horas]")
plt.ylabel("Temperatura [°C]")
plt.title("Temperatura Chimenea")

plt.subplot(313)
plt.plot(x_plot, altura_geografia, 'k')
plt.grid()
plt.title("Geografia de la Zona")
plt.xlabel("longitud [m]")
plt.ylabel("altura [m]")

plt.savefig("figuraCDB")
plt.tight_layout()  # no se solapan los graficos
plt.show()

# INICIAMOS LA GRILLA
Lx = 4000
Ly = 2000
Lt = 24

Nx = 400
Ny = 200
Nt = 500

hx = Lx/(Nx - 1)
hy = Ly/(Ny - 1)
ht = Lt/(Nt - 1)


# inicializamos grilla con los valores de temperatura
temperatura = np.zeros((Nx, Ny))


# CDB
# creamos el vector CDB para la superficie, para ver como se comporta en el
# tiempo
CDB_sup_plot = np.zeros((Nx, Nt))
for i in range(Nx):
    for j in range(Nt):
        # posicion horizontal i, tiempo j
        CDB_sup_plot[i, j] = CDB(i*hx, j*ht)


plt.figure(2)
plt.clf()
plt.imshow(CDB_sup_plot)
plt.colorbar()
plt.grid()
plt.show()
plt.title("Condiciones de Borde en el tiempo de la superficie")

# UBICAR CASILLAS QUE ESTAN EN LA SUPERFICIE
# primero encontramos la altura de cada CDB segun la GRILLA
# coleccion de indices de altura en la que se inicia la grilla hacia arriba
casilla_CDB = np.zeros(Nx)
for i in range(Nx):
    altura_CDB = geografia(i*hx)
    casilla_CDB[i] = int(altura_CDB/hy)


# FUNCION QUE ASIGNA CDB A TODA LA GRILLA EN FUNCION DEL TIEMPO
# solo esnecesario hacerlo para la superficie y los bordes
def iniciar_CDB(temperatura, k):
    """
    Implementa las condiciones de borde del problema, es decir, pone la
    temperatura del techo y los bordes segun temeratura_atmosfera() y la
    temperatura duperficial con CDB(), el cual contiene la chimenea, el mar
    y las temperaturas del suelo, todo en funcion de el paso k-esimo
    del tiempo y la grilla de temperatura

    temperatura: grilla que contiene las temperaturas espaciales
    k: paso de tiempo k-esimo

    retorna la grilla modificada con las condiciones de borde adecuadas
    """
    # CDB techo
    techo = temperatura_atmosfera(Ny*hy, k*ht)
    for i in range(Nx):
        temperatura[i, - 1] = techo

    # CDB bordes
    for j in range(Ny):
        borde = temperatura_atmosfera(j*hy, k*ht)
        temperatura[0, j] = borde
        temperatura[-1, j] = borde

    # CDB superficiales
    for i in range(Nx):
        CDB_sup = int(CDB(i*hx, k*ht))
        sup = int(casilla_CDB[i])
        for j in range(sup+1):
            temperatura[i, j] = CDB_sup

    return temperatura


def paso_relajacion(temperatura, w=w, k=1):
    """
    realiza un paso utilizando el metodo de la sobre-relajacion

    temperatura: grilla que contiene los valores de la temperatura en cada
    punto de la discretizacion
    w: parametro de sobre-relajacion, debe estar entre 0 y 2
    k: paso temporal k-esimo

    retorna:
    temperatura (actualizada)
    """
    # ponemos las condiciones de borde
    temperatura = iniciar_CDB(temperatura, k*ht)
    # actualizamos las casillas que estan en el interior
    # del borde definido por las CDB
    for i in range(1, Nx-1):
        # determinamos donde esta la CDB
        superficie = int(casilla_CDB[i])
        for j in range(superficie+1, Ny-1):
            temperatura[i, j] = (1-w)*temperatura[i, j] + w/4*(
                               temperatura[i+1, j] +
                               temperatura[i-1, j] +
                               temperatura[i,   j+1] +
                               temperatura[i,   j-1]
                               )
    return temperatura

# reiniciamos temperatura, para que no hay solapamiento accidental
temperatura = np.zeros((Nx, Ny))
archive = np.zeros((Nx, Ny, Nt))

# INICIAMOS ATMOSFERA
# ponemos las CDB de la atmosfera
for i in range(Nx):
    for j in range(Ny):
        temperatura[i, j] = temperatura_atmosfera(j*hy, 0)
temperatura = iniciar_CDB(temperatura, 0)

for i in range(Nt):
    archive[:, :, i] = temperatura
    temperatura = paso_relajacion(temperatura, w=w, k=i*ht)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

plt.figure(3)
plt.clf()
plt.imshow(np.transpose(temperatura), origin="lower", extent=[0, Lx, 0, Ly])
plt.colorbar()
plt.grid()
plt.title('Temperatura Final t = {} y w = {}'.format(Lt, w))
plt.xlabel("Longitud [m]")
plt.ylabel("Altura[m]")
plt.show()

plt.figure(4)
plt.clf()

x = np.linspace(0, 4000, 10000)
y = np.zeros(len(x))

for i in range(len(x)):
    y[i] = geografia(x[i])

# t = 0
plt.subplot(231)
plt.imshow(
    (np.transpose(archive[:, :, int(0)])), origin="lower", vmin=-20,
    norm=SymLogNorm(linthresh=1, vmin=-15, vmax=3000),
    extent=[0, 4000, 0, 2000])
plt.plot(x, y, 'k')
plt.title("Temperatura t = 0 [h]")
plt.xlabel("Distancia horizontal [m]")
plt.ylabel("Altura [m]")
plt.grid()

# t = 8
plt.subplot(232)
plt.imshow(np.transpose(archive[:, :, int(Nt/3)]), origin="lower", vmin=-20,
           norm=SymLogNorm(linthresh=1, vmin=-15, vmax=1500),
           extent=[0, 4000, 0, 2000])
plt.plot(x, y, 'k')
plt.title("Temperatura t = 8 [h]")
plt.xlabel("Distancia horizontal [m]")
plt.grid()

# t = 12
plt.subplot(233)
plt.imshow(
    np.transpose(archive[:, :, int(Nt/2)]), origin="lower", vmin=-20,
    norm=SymLogNorm(linthresh=1, vmin=-15, vmax=1500),
    extent=[0, 4000, 0, 2000])
plt.plot(x, y, 'k')
plt.title("Temperatura t = 12 [h]")
plt.xlabel("Distancia horizontal [m]")
plt.grid()

# t = 16
plt.subplot(234)
plt.imshow(
   np.transpose(archive[:, :, int(Nt/3*2)]), origin="lower", vmin=-20,
   norm=SymLogNorm(linthresh=1, vmin=-15, vmax=1500),
   extent=[0, 4000, 0, 2000])
plt.plot(x, y, 'k')
plt.title("Temperatura t = 16 [h]")
plt.xlabel("Distancia horizontal [m]")
plt.ylabel("Altura [m]")
plt.grid()

# t = 20
plt.subplot(235)
plt.imshow(np.transpose(archive[:, :, int(Nt/6*5)]), origin="lower", vmin=-20,
           norm=SymLogNorm(linthresh=1, vmax=1500),
           extent=[0, 4000, 0, 2000])
plt.plot(x, y, 'k')
plt.title("Temperatura t = 20 [h]")
plt.xlabel("Distancia horizontal [m]")
plt.grid()

# t = 24
plt.subplot(236)
plt.imshow(np.transpose(archive[:, :, int(Nt-1)]), origin="lower", vmin=-20,
           norm=SymLogNorm(linthresh=1, vmax=1500),
           extent=[0, 4000, 0, 2000])
plt.plot(x, y, 'k')
plt.title("Temperatura t = 24 [h]")
plt.xlabel("Distancia horizontal [m]")
plt.grid()

# colorbar
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.05, 0.8])
clb = plt.colorbar(cax=cax)
clb.set_label("Temperatura")
plt.legend()

plt.suptitle('Evolución Temporal de la Temperatura')

plt.show()
