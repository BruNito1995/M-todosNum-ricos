#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math


G = 6.674e-11 # constante de graviración universal
M = 1e13 # masa sol
m = 1/G/M

class Planeta(object):
    """
    Complete el docstring.
    """

    def __init__(self, condicion_inicial, alpha=0):
        """
        __init__ es un método especial que se usa para inicializar las
        instancias de una clase.

        Ej. de uso:
        >> mercurio = Planeta([x0, y0, vx0, vy0])
        >> print(mercurio.alpha)
        >> 0.
        """
        self.y_actual = condicion_inicial
        self.t_actual = 0.
        self.alpha = alpha

    def ecuacion_de_movimiento(self,t):
        """
        Implementa la ecuación de movimiento, como sistema de ecuaciones de
        primer orden.
        El planeta se rige por la segunda ley de Newton en donde asumiremos
        que la unica fuerza que lo afecta es la atraccion gravitatoria
        del sol:

        F = ma = GmM/r^2

        Para obtener el sistema de ecuaciones se resuelve descomponiendo
        en los ejes cartesianos y resolviendo por separado

        Fx = mx'' = GmM/r^2*cos(theta)
        Fy = my'' = GmM/r^2*sin(theta)
        dx/dt = vx
        dy/dt = vy

        Donde r es la distancia al sol y theta el angulo
        que barre el vector posicion
        """
        x, y, vx, vy = self.y_actual
        theta = math.atan(y/x)
        r = (x**2 + y**2)**0.5
        fx = G*M*m/r**2*math.cos(theta)
        fy = G*M*m/r**2*math.sin(theta)
        return [vx, vy, fx, fy]

    def avanza_rk4(self, dt):
        """
        Toma la condición actual del planeta y avanza su posición y velocidad
        en un intervalo de tiempo dt usando el método de RK4. El método no
        retorna nada, pero modifica los valores de self.y_actual.
        """


        pass

    def avanza_verlet(self, dt):
        """
        Similar a avanza_rk4, pero usando Verlet.
        """
        pass

    def energia_total(self):
        """
        Calcula la energía total del sistema en las condiciones actuales.
        Energia = Potencial gravitatoria + Cinetica
        Potencial = -GmM/r + alpha GMm/r^2
        Cinetica = mv^2/2
        """
        x, y, vx, vy = self.y_actual
        alpha = self.alpha

        r = (x**2 + y**2)**0.5
        v = (vx**2 + vy**2)**0.5

        energia_cinetica = m*v**2*0.5
        energia_potencial = -G*M*m/r + alpha*G*M*m*r**2
        energia_total = energia_cinetica + energia_potencial

        return energia_total
