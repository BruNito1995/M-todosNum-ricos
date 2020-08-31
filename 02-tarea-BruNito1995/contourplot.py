#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Este script grafica la función:

f(x, y) = 10 * (np.exp(-x**2 - y**2) - np.exp(-((x-1)/1.5)**2 - ((y-1)/0.5)**2))

Y agrega lineas de contorno en los niveles [-6, -4, -2, 0, 2, 4, 6]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

x = np.linspace(-3, 3, 300)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)
Z = 10 * (np.exp(-X**2 - Y**2) - np.exp(-((X-1)/1.5)**2 - ((Y-1)/0.5)**2))

plt.clf()
plt.imshow(Z, origin='lower', cmap=cm.viridis, extent=(-3, 3, -2, 2))

levels = [-6, -4, -2, 0, 2, 4, 6]
plt.contour(X, Y, Z, levels)

plt.xlabel('$x$', fontsize=18)
plt.ylabel('$y$', fontsize=18)
title = (r'$10 \times (\exp(-x^2 - y^2) - \exp((\frac{x-1}{1.5})^2 - '
         r'(\frac{y-1}{0.5})^2))$')
plt.suptitle(title, fontsize=18)

# plt.show()
plt.savefig('contourplot.png')
