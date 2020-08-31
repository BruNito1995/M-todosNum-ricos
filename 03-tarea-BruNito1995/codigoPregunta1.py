import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spl

plt.close('all')

def funcion1(x):
    """
    Calcula el valor de la funcion del enunciado
    """
    valor = 1/(1+25*x**2)
    return valor

def lagrange(x,x_val,y_val):
    """
    calcular el valor de el polinomio interpolado con los
    datos val_x y val_y en el punto x, a traves del metodo
    de lagrangre
    """

    k = len(x_val)

    #definimos las funciones bases
    def base(j):
        p = 1
        for i in range(k):
            if i !=j:
                p *= (x - x_val[i])/(x_val[j] - x_val[i])
        return p

    valor = 0

    for i in range(k):
        valor += base(i)*y_val[i]

    return valor

def pol_lagrange(x_val,y_val):
    """
    funcion de lagrange mejorada, retorna una funcion con
    el polinomio de Lagrenge que ajusta los puntos
    x_val e y_val entregados
    """
    k = len(x_val)

    def base(x,j):
        p = 1
        for i in range(k):
            if i !=j:
                p *= (x - x_val[i])/(x_val[j] - x_val[i])
        return p

    def pol(x):
        valor = 0
        for i in range(k):
            valor += base(x,i)*y_val[i]
        return valor

    return pol

#probamos
# n es el número de puntos
n = 100
valx = np.linspace(-1,1,n)
valy = []
ajuste_pol = []
ajuste_spline = []

#definimos el vector valy como y = f(x)
for i in range(len(valx)):
    valy.append(funcion1(valx[i]))

polinomio = pol_lagrange(valx,valy)
spline = spl.InterpolatedUnivariateSpline(valx,valy, k = 5)

#m es el numero de puntos para el ajuste
#de Lagrange
m = n+100
x_ajuste = np.linspace(-1,1,m)
for i in range(len(x_ajuste)):
     ajuste_pol.append(polinomio(x_ajuste[i]))
     ajuste_spline.append(spline(x_ajuste[i]))

print("polinomio en 1 = ", polinomio(1))
print("polinomio en 1.3 = " ,polinomio(1.3))

def aux(x):
    """
    funcion auxiliar para estudiar el comportamiento de funcines
    que retornan funciones retorna una funcion aux2(y) = x^y
    """
    def aux2(y):
        return x**y

    return aux2

#mostramos polinomio
plt.subplot(2,1,2)
plt.plot(valx,valy,'ro', label=r'Muestreo $f(x)$')
plt.plot(x_ajuste,ajuste_pol, label = "Polinomio de Lagrange")
plt.legend()
plt.ylabel("f(x)");plt.xlabel("x")


#mostramos spline
plt.subplot(2,1,1)
plt.title("Ajuste de polinomio de Lagrange y Spline con N = " + str(n))
plt.plot(valx,valy,'ro', label=r'Muestreo $f(x)$')
plt.plot(x_ajuste,ajuste_spline, label = "Spline")
plt.ylabel("f(x)")
plt.legend()
plt.savefig('ajuste'+str(n)+'.png')

plt.show()
