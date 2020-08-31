
# -*- coding: utf-8 -*-
"""
Se deriva la función coseno evaluada en x = 1.271, y se comparan 2 algoritmos,
con errores de orden 1 y 4. Se comparan los resultados de la derivada numerica
con el valor arrojado por math.sin(x)

"""
#importamos las librerias que ocuparemos
import math as math
import numpy as np
import matplotlib.pyplot as plt

def derivada_o4(func,x,h=0.01):

    """
    Funncion que deriva numericamente una funcion entregada, evaluada en el
    punto x y con diferencial h. Tiene un error de orden de h^4
    """

    f = func
    derivada = np.float64(
        (-f(x+2*h) + 8*f(x+h) - 8*f(x-h) + f(x-2*h))/12/h
        )
    return derivada

def derivada_o1(func,x,h=0.01):

    """
    Funncion que deriva numericamente una funcion entregada, evaluada en el
    punto x y con diferencial h. Tiene un error de orden de h^1

    """
    f = func
    derivada = np.float64((f(x+h)-f(x))/h)
    return derivada

#punto en el que se deriva
x = np.float64(1.271)

#valor de referencia
valor_real = -math.sin(x)

#vector de valores para h
h = np.logspace(-1.0, -21.0, num=1000)

#recordar dejar los numeros como float
derivada_precisa = np.array([0.0]*len(h))
derivada_imprecisa = np.array([0.0]*len(h))

for i in range(len(h)):
    #evaluamos para cada h
    #ojo! las funciones se entregan sin argumento dentro al ser llamadas como argumentos
    derivada_precisa[i] = derivada_o4(math.cos,x,h[i])
    derivada_imprecisa[i] = derivada_o1(math.cos,x,h[i])

#determinamos las diferencias entre el valor real de la derivada y el
#calculado numericamente
diferencia_precisa = np.absolute(derivada_precisa - valor_real)
diferencia_imprecisa = np.absolute(derivada_imprecisa - valor_real)


#clf = clearfigures
plt.clf()
plt.loglog(h,diferencia_precisa, label='diferencia precisa' )
plt.loglog(h,diferencia_imprecisa, label ='diferencia imprecisa')
plt.xlabel('h',fontsize = 15)
plt.ylabel('error',fontsize = 15)
plt.legend()
plt.title('Diferencia entre algoritmos derivadores, con float 32',fontsize=12)

#guardamos la figura
plt.savefig('Derivadas_32.png')
plt.show()

minimp = min(diferencia_imprecisa)
minpre = min(diferencia_precisa)

print('El mejor ajuste tiene un error de', minimp,
'mientras que para el peor es de ' , minpre)
