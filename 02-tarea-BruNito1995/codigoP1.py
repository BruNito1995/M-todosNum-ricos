"""
Codigo para resolver la segunda tarea de Metodos Numéricos
P1
"""
import math as math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as int


plt.close('all')

#valores de enunciado
separacion = 20
caida = 7.5
x0 = separacion/2

def catenaria(a,x=10,x0=10):
    """
    Calcula el valor de la funcion catenaria
    """
    exp = math.exp
    exponente = (x-x0)/a
    valor = a/2*(exp(exponente) + exp(-exponente))
    return valor

def dcatenaria(a,x=10,x0=10):
    """
    Calcula el valor de la derivada con respecto
    a x de la catenaria en el punto x
    """
    exp = math.exp
    c = (x-x0)/a
    valor = 0.5*(c*exp(c) - c*exp(-c))
    return valor

def dcatenaria_a(a,x=10,x0 = 10):
    """
    calcula la derivada de la catenaria con respecto a
    alpha
    """
    c = (x-x0)/a
    exp = math.exp
    valor = 0.5*(exp(c)*(1-c) + exp(-c)*(c+1))
    return valor

def integrando(x,a,x0):
    """
    calcula el integrando para calcular
    el largo del cable
    """
    valor = (dcatenaria(a,x,x0)**2+1)**0.5
    return valor


#ahora hacemos el algoritmo de la biseccion
#usamos biseccion porque no permite
#definir el intervalo que usa el algoritmos

def biseccion(func,a,b,error=1e-12):
    """
    Calcula la raiz de la funcion entre a y b con una tolerancia de
    error, a traves del metodo de la biseccion
    """

    if func(a)*func(b) > 0:
        return "valores de a y b INVALIDOS"
    p = (a+b)*0.5
    while np.absolute(func(p)) > error:
        p = (a+b)*0.5
        if func(a)*func(p)<0:
            b=p
        elif func(a)*func(p)>0:
            a=p
        else:
            return p
    return p

def newton(func,dfunc,x,error=1e-12):
    """
    calcula el cero de la funcion utilizando el metodo de
    newton, con x el pnto de partida, error es la tolerancia y func
    con dfun son la funcion y su derivada
    """

    while np.absolute(func(x))>error:
        x = x - func(x)/dfunc(x)

    return x




def h(x):
    """
    funcion auxiliar para encontrar ceros apropiados
    """
    valor = catenaria(x,10,10) - catenaria(x,0,10) -7.5
    return valor

def dh(x):
    """
    derivada de h conrespecto a alpha
    """
    valor = dcatenaria_a(x,10,10) - dcatenaria_a(x,0,10)
    return valor

cero_newton = newton(h,dh,-5)
print("El metodo de newton converge a alpha = ",cero_newton)

cero_biseccion = biseccion(h,-5,-8)
print("la biseccion converge a: " , cero_biseccion)
#encontramos el valor de a
a = cero_biseccion
#ahora debemos integrar el largo para determinar
#cuanto cable se necesita

#INTEGRAL(dcatenria(a),0,20)
largo_cable = int.quad(integrando,0,20, args=(a,x0))[0]
print("el largo del cable debe ser: ", largo_cable, " metros")

#grafico catenaria
xcat = np.linspace(0, 20, 300)
ycat = []

for i in range(0,len(xcat)):
    #ycat.append() = catenaria(x = xcat[i], a = -7.5, x0 = 10)
    ycat.append(catenaria(x = xcat[i], a=7.5,x0 = 10)-7.5)

plt.hlines(0,0,20, label=" y = 0")
plt.plot(xcat,ycat, label="catenaria")
plt.xlabel("x [metros]")
plt.ylabel("Altura [metros]")
plt.title(r"Catenaria del cable con $\alpha = -7.5$")
plt.legend()
plt.savefig('catenaria.png')
plt.show()
