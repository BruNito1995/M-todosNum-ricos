"""
En este archivo se resuelve la pregunta 1 de la tarea 9 de metodos numericos.
Se solicita encontrar la posicion del centro de masa de un solido que esta
descrito por la interseccion de un toroide y un cilindro de ecuaciones, donde
la densidad es variable

Toro: z^2 + ( (x^2 + y^2)^0.5 - 3)^2 <= 1
Cilindro: (x - 2)^2 + z^2 <= 1

Densidad: p(x, y, z) = 0.5*(x^2 + y^2 + z^2)
Para ello se utilizara un metodo de Monte Carlo
"""
import random

random.seed(1278912317) # para mantener resultados replicables
Ntot = 1e6

def densidad(x, y, z):
    """
    define la densidad de un punto en el volumen
    """
    d = 0.5*(x**2 + y**2 + z**2)
    return d


def toroide(x, y, z):
    """
    Define si un punto se encuentra dentro del toroide
    """

    t = z**2 + ((x**2 + y**2)**0.5 - 3)**2

    if t > 1:
        return False
    else:
        return True


def cilindro(x, y, z):
    """
    Define si un punto se encuentra dentro del Cilindro
    """
    c = (x - 2)**2 + z**2

    if c > 1:
        return False
    else:
        return True


def en_volumen(x, y, z):
    """
    Define si un punto esta dentro del volumen estudiado, va a evaluar si se
    encuentra simultaneamente en el cilindro y en el toroide
    """
    # esta en en volumen => esta en cilindro Y en toroide
    if toroide(x, y, z) and cilindro(x, y, z):
        return True

    else:
        return False

# el toroide cabe en una caja de 4x4x2

def centro_masa(Ntot = Ntot):
    """
    calcua el centro de masa de un volumen generado por la interseccion de un
    cono y un toroide (definido en en_volumen()) con densidad definida por la
    funcion densidad().
    Se calcula utilizando un metodo de integracion de montecarlo

    input:
    Ntot = numero de iteraciones

    output:
    x_cm, y_cm, z_cm : coordenadas (x, y, z) del centro de masa
    dx, dy, dz : estimacion del intervalo de confianza para cada coordenada del
    centro de masa
    """

    # inicializamos las variables
    xmin, xmax = -2,2
    ymin, ymax = -2,2
    zmin, zmax = -1,1
    vol = (-xmin + xmax)*(-ymin + ymax)*(-zmin + zmax)

    counter = 0
    x_cm, varx = 0, 0
    y_cm, vary = 0, 0
    z_cm, varz = 0, 0
    m, varm = 0, 0

    while counter < Ntot:

        # definir puntos aleatorios
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        z = random.uniform(zmin, zmax)

        counter += 1

        # ver si esta contenido en el en el volumen
        # si esta contenido, sumar a las cordenadas del centro de masa
        if en_volumen(x,y,z):
            d = densidad(x,y,z)

            x_cm += x*d
            y_cm += y*d
            z_cm += z*d

            # masa total
            m += d

            # desviaciones
            varx += (x*d)**2
            vary += (y*d)**2
            varz += (z*d)**2
            varm += d**2

        # si no esta contenido volver al sgte punto
        else:
            pass
    # calculamos centro de masa
    x_cm = x_cm/m
    y_cm = y_cm/m
    z_cm = z_cm/m

    dx=vol*((varx/Ntot-( x_cm/Ntot )**2)/Ntot)**0.5
    dy=vol*((vary/Ntot-( y_cm/Ntot )**2)/Ntot)**0.5
    dz=vol*((varz/Ntot-( z_cm/Ntot )**2)/Ntot)**0.5
    dm = vol*((varm/Ntot - (m/Ntot)**2)/Ntot)**0.5

    return x_cm, y_cm, z_cm, dx, dy, dz

if __name__ == "__main__":
    x_cm, y_cm, z_cm, dx, dy, dz = centro_masa()
    a ="""
    Se estima que el centro de masa se encuentra en las coordenadas:
    x = {:06.4f} +- {:06.4f}
    y = {:06.4f} +- {:06.4f}
    z = {:06.4f} +- {:06.4f}
    Despues de {} iteraciones
    """.format(x_cm, y_cm, z_cm, dx, dy, dz, Ntot)

    print(a)
