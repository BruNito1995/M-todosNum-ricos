"""
En este archivo se resuelve la tarea 12 de metodos numericos, la cual consiste
en realizar analisis bayesianos para estimar los parametros de un modelo.
Se utilizaran las temperaturas promedio anuales de los datos entregados y se
estimara un modelo para esta variable
"""
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit import (minimize, Parameters, report_fit)
import scipy.stats

data = np.genfromtxt('GLB.txt')

# limpiamos Nan, considerando el promedio de los meses
# adyacentes
for i in range(139):
    for j in range(18):
        if np.isnan(data[i, j]):
            data[i, j] = data[i, j - 1] + data[i, j+1]

# manualmente ajustamos los nan que quedan en noviembre y diciembre
data[-1, 12] = data[-2, 12]
data[-1, 11] = data[-2, 11]
data[-1, 13] = data[-2, 13]

años = data[:, 0]
promedio = data[:, 13]


def likelihood(x_data, y_data, params, modelo):
    '''
    A partir de ciertos datos y algun modelo, calcula su verosimilitud
    '''
    N = len(x)
    S = np.sum((y_data - modelo(x_data, *params))**2)
    L = (2 * np.pi * 1.5**2)**(-N / 2.) * np.exp(-S / 2 / 1.5**2)
    return L

def prior(beta, params):
    '''
    Dada una distribucion a priori de los parametros, retorna la probabilidad
    de cierto modelo en un punto

    input
    beta:
    params:

    P: la probabilidad de el modelo, considerando una distribucion de los
    parametros
    '''
    beta0, beta1, beta2 = beta
    mu0, sigma0, mu1, sigma1, mu2, sigma2 = params
    S0 = -1. / 2 * (beta0 - mu0)**2 / sigma0**2
    S1 = -1. / 2 * (beta1 - mu1)**2 / sigma1**2
    S2 = -1. / 2 * (beta2 - mu2)**2 / sigma2**2
    P = np.exp(S0 + S1 + S2 + S3)
    P = P / (2 * np.pi * sigma0 * sigma1 * sigma2 * sigma3)

return P

def fill_prior_1(beta0_grid, beta1_grid, prior_params, data):
    """
    dada una grilla de valores para los beta, los parametros a priori y los
    datos, calcula la desnsidad dde probabilidad y la verosimilitud
    """
    output = np.zeros(beta0_grid.shape)
    ni, nj = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            output[i, j] = prior([beta0_grid[i, j], beta1_grid[i, j]],
                                 prior_params)
            likelihood_m1[i, j] = likelihood([data[0], data[1]],
                                             [beta0_grid[i, j],
                                              beta1_grid[i, j]], modelo_1)
return output, likelihood_m1
