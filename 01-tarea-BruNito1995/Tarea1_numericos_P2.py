import numpy as np
import math as math
import matplotlib.pyplot as plt
import scipy.integrate

plt.close('all')
fname = 'firas_monopole_spec_v1.txt'

#importamos el texto
texto = np.loadtxt(fname)
# INFORMACION DEL ARCHIVO
# Column 1 = frequency from Table 4 of Fixsen et al., units = cm^-1
# Column 2 = FIRAS monopole spectrum computed as the sum
#             of a 2.725 K BB spectrum and the
#             residual in column 3, units = MJy/sr
# Column 3 = residual monopole spectrum from Table 4 of Fixsen et al.,
#             units = kJy/sr
# Column 4 = spectrum uncertainty (1-sigma) from Table 4 of Fixsen et al.,
#             units = kJy/sr
# Column 5 = modeled Galaxy spectrum at the Galactic poles from Table 4 of
#             Fixsen et al., units = kJy/sr
#

frecuency = texto[:,0]*100*2.99e8#en c m^-1
monopoleSpec = texto[:,1]*1e-20 #en Ws/m2 /sr
residualSpec = texto[:,2] 
uncertainity = texto[:,3]*1e-17 #en Ws/m2 /sr
galaxySpectrum = texto[:,4] # en kJy/s

fig, ax = plt.subplots()
ax.errorbar(frecuency,monopoleSpec,yerr = uncertainity*1e-3, ecolor="r")
plt.ylabel(r"Espectro $[\frac{Ws}{m^2 sr}]$")
plt.xlabel(r"Frecuencia [Hz]")

plt.title("Espectro de radiacion de fondo de microondas",fontsize=12)
plt.savefig('Radiaciondefondo.png')



def integracion(func,inicio,fin,dx=0.01):
    """
    calcula la integral de la funcion entregada
    entre inicio y fin, a traves de una suma de Riemann
    con un diferencial dx
    """
    integral = 0
    rango_integracion = np.arange(inicio,fin,dx)
    for i in rango_integracion:
        integral =(
            integral + func(i)*dx
        )
    return integral

# para probar integracion 
def cuadrado(x):
    """
    Calcula el cuadrado del valor entregado
    """
    return x*x

def plankAux(x):
    """
    funcion auxiliar para calcular la integral de plank,
    utilizando el cambio de variables y = arctan(x).
    """
    #para facilitar la notacion
    tan = math.tan
    exp = math.exp
    
    valor = tan(x)**3*(1+tan(x)**2)/(exp(tan(x))-1)
    return valor

tolerancia = 1e-5
valor_esperado = math.pi**4/15
dx = 0.5
error = tolerancia + 1

#integramos la funcion plankaux y mejoramos el valor hasta cumplir con 
#la tolerancia
#INTEGRAL 2
while np.absolute(error) > tolerancia:
    #mejorar integracion
    integral = integracion(plankAux,0.001,math.pi*0.5,dx)
    error = integral - valor_esperado
    dx = dx/2

print(
    'error de calculo de integral de Planck con integracion propia=', np.absolute(error))

#COMPARACION CON SCIPY
quad = scipy.integrate.quad(plankAux,0.01,math.pi*0.5-0.01)[0]
error2 = np.absolute(quad - valor_esperado)

print('error de calculo de integral de Planck con integracion modulo integrate.quad = ', error2)

#parametros para simpson
deltaFrecuency = frecuency[2] - frecuency[1] #dx
#PARES IMPARES Y BORDES
evenSpectrum = monopoleSpec[2:len(monopoleSpec)-2:2]#asegurarse de que no se incluya el ultimo termino
oddSpectrum = monopoleSpec[1:len(monopoleSpec)-2:2]
edgeSpectrum = monopoleSpec[0] + monopoleSpec[len(monopoleSpec)-1]

#entonces segun simpson el area es 
# INTEGRAL 3
area_bajo_espectro = deltaFrecuency/3*(edgeSpectrum + 4*sum(oddSpectrum) + 2*sum(evenSpectrum))
area_bajo_espectro2 = np.trapz(monopoleSpec, x=frecuency)
print('Area bajo espectro, por metodo de Simpson = ', area_bajo_espectro)
print('Area bajo espectro, por trapz de numpy = ', area_bajo_espectro2)

h = 6.62607004e-34
c = 2.99792458*1e8
kb = 1.38064852e-23

def plank(T,f):
    """
    calcula el valor de la funcion de plank
    en funcion de la temperatura en K y frecuencia
    """
    exp = math.exp
    num = 2*h*f**3/c**2
    exponente = h*f/(kb*T)
    valor = num / (exp(exponente)-1)
    
    return valor

temperatura = (area_bajo_espectro*c**2*h**3*15/2/kb**4/math.pi**4)**(1/4)
print('Se calcula una temperatura de ', temperatura, ' K')

temperatura_real = 2.725

#generamos los nuevos vectores de la
#funcion de plank
plank_real = np.zeros(len(frecuency))
plank_propio = np.zeros(len(frecuency))

for i in range(0,len(frecuency)):
    plank_real[i] = plank(temperatura_real,frecuency[i])
    plank_propio[i] = plank(temperatura, frecuency[i]) 

#generamos los nuevos graficos

plt.figure(2)
plt.plot(frecuency,plank_real, label="Funcion de Plank, temperatura real")
plt.plot(frecuency,plank_propio, label ="Funcion de Plank, temperatura calculada")
plt.plot(frecuency,monopoleSpec, label="Radiacion de fondo FIRAS")
plt.legend()
plt.xlabel(r"Frecuencia $\frac{1}{m}$")
plt.ylabel(r"Espectro $[\frac{Ws}{m^2 sr}]$")
plt.title("Comparación entre modelos")
plt.savefig('comparacion.png')
plt.show()