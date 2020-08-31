"""
codigo P2 metodos numericos
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

plt.close('all')

def funcion1(x,y):
    """
    funcion F1 definida en el enunciado
    """
    valor = x**4 + y**4 -15

    return valor

def funcion2(x,y):
    """
    funcion f2 definida en el enunciado
    """
    valor = x**3*y - x*y**3 - y/2 - 1.271
    return valor

#generamos grafico
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)
#Z = 10 * (np.exp(-X**2 - Y**2) - np.exp(-((X-1)/1.5)**2 - ((Y-1)/0.5)**2))
Z = funcion1(X,Y)
plt.figure(1)
plt.imshow(Z, origin='lower', cmap=cm.viridis, extent=(-2, 2, -2, 2))

#levels = [-6, -4, -2, 0, 2, 4, 6]
levels = [0]
plt.contour(X, Y, Z, levels)

plt.xlabel('$x$', fontsize=18)
plt.ylabel('$y$', fontsize=18)
title = ('F1(x,y)')
plt.suptitle(title, fontsize=18)
plt.savefig('contornoF1.png')
#plt.show()


#INICIO PARAMETRIZACION
r = 15**0.5

def curva(t):
    """
    define la curva que parametriza los ceros de la funnción
    funcion1(x,y), como un ciculoide utilizando el cambio de
    variables
    x = sign(sin)*(sin^2*r^2)^1/4
    y = sign(cos)*(cos^2*r^2)^1/4
    """
    #parametrizamos
    x = (np.sin(t)**2*r**2)**0.25*(np.sign(np.sin(t)))
    y = (np.cos(t)**2*r**2)**0.25*(np.sign(np.cos(t)))

    return [x,y]


t = np.linspace(0,np.pi*2,500)
#graficamos la parametrizacion
[X,Y] = curva(t)
plt.figure(2)
plt.plot(X,Y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Parametrización ceros de F1(x,y)")
plt.savefig('parametrizacion.png')


#ahora definimos la biseccion modificada
def biseccion_modificada(curva,func,a,b, error = 1e-7):
    """
    Utiliza el metodo de la biseccion para encontrar
    ceros, pero recorre una curva arbitraria
    en vez de un eje
    curva = curva que recorre el metodo
    func = funcion a la que se le buscan los ceros
    a y b = intervalo que inicia la biseccion
    error = tolerancia
    """
    
    def para(x):
        return func(curva(x)[0],curva(x)[1])


    if para(a)*para(b) > 0:
        return "valores de a y b INVALIDOS"
    p = (a+b)*0.5
    while np.absolute(para(p)) > error:
        p = (a+b)*0.5
        if para(a)*para(p)<0:
            b=p
        elif para(a)*para(p)>0:
            a=p
        else:
            return p
    return p

# para encontrar a y b apropiados graficamos F2
# en la curva parametrizada
plt.figure(3)
plt.plot(t,funcion2(X,Y))
plt.xlabel(r"$\theta [rad]$")
plt.ylabel(r"$F2")
plt.title("Función 2 en curva parametrizada")
plt.hlines(0,0,np.pi*2, label=" y = 0")
plt.legend()
plt.savefig('cerosf2.png')


#a partir del grafico definimos los a y b adecuados
#para cada 0 de la curva parametrizada
[a1,b1] = [0.7 , 1.2]
[a2,b2] = [1.4 , 1.7]
[a3,b3] = [2.1 , 2.6]
[a4,b4] = [2.9 , 3.3]
[a5,b5] = [3.7 ,  4.1]
[a6,b6] = [4.5 , 4.8]
[a7,b7] = [5.5 , 5.7]
[a8,b8] = [5.9 , 6.27]

#buscamos lo 8 ceros de la funcion2 sobre la parametrizacion
cero1 = biseccion_modificada(curva,funcion2,a1,b1)
cero2 = biseccion_modificada(curva,funcion2,a2,b2)
cero3 = biseccion_modificada(curva,funcion2,a3,b3)
cero4 = biseccion_modificada(curva,funcion2,a4,b4)
cero5 = biseccion_modificada(curva,funcion2,a5,b5)
cero6 = biseccion_modificada(curva,funcion2,a6,b6)
cero7 = biseccion_modificada(curva,funcion2,a7,b7)
cero8 = biseccion_modificada(curva,funcion2,a8,b8)

ajuste = [funcion2(curva(cero1)[0],curva(cero1)[1]),
          funcion2(curva(cero2)[0],curva(cero2)[1]),
          funcion2(curva(cero3)[0],curva(cero3)[1]),
          funcion2(curva(cero4)[0],curva(cero4)[1]),
          funcion2(curva(cero5)[0],curva(cero5)[1]),
          funcion2(curva(cero6)[0],curva(cero6)[1]),
          funcion2(curva(cero7)[0],curva(cero7)[1]),
          funcion2(curva(cero8)[0],curva(cero8)[1])]

plt.figure(4)
plt.grid()
plt.plot(np.absolute(ajuste),'ro')
plt.title("Valor de la funcion F2 en los ceros encontrados")
plt.xlabel("cero")
plt.ylabel("F2(cero)")
plt.savefig('ajuste.png')

plt.show()
