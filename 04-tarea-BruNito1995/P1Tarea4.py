import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.linalg import inv
from scipy.linalg import lu
from scipy.linalg import solve_triangular

plt.close("all")
g = 9.81

P = 4.271*g
P1 = 1.5*g #parámetro libre
P2 = 1*g

#definimos las proyecciones sobre los ejes necesarias
pi = math.pi
x1 = math.sin(pi/4)
x2 = math.sin(pi/6)
x3 = math.sin(pi/12)
y1 = math.cos(pi/4)
y2 = math.cos(pi/6)
y3 = math.cos(pi/12)

A = np.matrix([
    [-x1,x1, 0, 0, 0, 0,x2, 0, 0, 0, 0, 0],#nodo1x
    [y1,y1, 0, 0, 0, 0,-y2, 0, 0, 0, 0, 0],#nodo1y
    [ 0, 0,-x1,x1, 0, 0, 0,-x2,x2, 0, 0, 0],#nodo2x
    [ 0, 0,y1,y1, 0, 0, 0,-y2,-y2, 0, 0, 0],#nodo2y
    [ 0, 0, 0, 0,-x1,x1, 0, 0, 0,-x2, 0, 0],#nodo3x
    [ 0, 0, 0, 0,y1,y1, 0, 0, 0,-y2, 0, 0],#nodo3y
    [ 0, 0, 0, 0, 0, 0,-x2,x2, 0, 0,x3, 0],#nodo4x
    [ 0, 0, 0, 0, 0, 0,y2,y2, 0, 0,-y3, 0],#nodo4y
    [ 0, 0, 0, 0, 0, 0, 0, 0,-x2,x2, 0,-x3],#nodo5x
    [ 0, 0, 0, 0, 0, 0, 0, 0,y2,y2, 0,-y3],#nodo5y
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-x3,x3],#nodo6x
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,y3,y3]#nodo7y
    ])


B = np.matrix([0,0,0,0,0,0,0,P1,0,P2,0,P])
B = B.transpose()

## Primera resolución
#solve de scipy.linalg
T1 = la.solve(A,B)
title = "Resolución del sistema con M1 =" + str(P1/g) + "[kg]"
maxima_tension = np.amax(T1)
print("La tensión máxima para M1 = " +
      str(P1/g) + "[kg] es de " + str(maxima_tension) +
      " Newton"
      )

##Segunda resolución
#descomposición LU
p, l , u = lu(A)
y = solve_triangular(l, p @ B, lower=True)
T2 = solve_triangular(u, y, lower=False)

#Tercera Resolución
#invertir A
invA = inv(A)
T3 = invA * B

#Comportamiento de T_max en función de M1
#iniciamos las variables y el rango de P1
P1_variable = np.linspace(0,2,10)*g
T_max = np.zeros((len(P1_variable),1))
cuerda_max = np.zeros((len(P1_variable),1))
for i in range(len(P1_variable)):
    B = np.matrix([0,0,0,0,0,0,0,P1_variable[i],0,P2,0,P]).transpose()
    sol = invA*B
    T_max[i] = (sol).max()
    cuerda_max[i] = np.where(sol == T_max[i])[0]

#graficos

#distribución tensiones
plt.figure()
plt.bar([1,2,3,4,5,6,7,8,9,10,11,12],T1[:,0])
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.title("Distribución de Tensiones para M1 =" +
            str(P1/g) + "[kg]")
plt.xlabel("Cuerda");plt.ylabel("Tensión [N]")
plt.grid()
plt.savefig("dist_tensiones.png")


#tensiones en función de M1 y cuerda que tiene la maxima tension
plt.figure()
#tensiones
plt.subplot(2,1,1)
plt.bar(P1_variable/g,T_max[:,0], width = 0.1)
plt.xlabel("M1 [kg]"); plt.ylabel("Tensión máxima [N]")
plt.ylim((20,30))
plt.grid()
plt.title("Comportamiento de la tensión máxima en función de M1")

# indices
plt.subplot(2,1,2)
plt.plot(P1_variable/g,cuerda_max+1, 'o')
plt.ylabel("Cuerda de Máxima Tensión");plt.xlabel('M1 [kg]')
plt.savefig("tensionesmaximas.png")

plt.show()
