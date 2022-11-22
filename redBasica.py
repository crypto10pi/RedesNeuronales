#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 19:24:47 2022

@author: jorge
"""

import numpy as np
import pandas as pd

# compute sigmoid nonlinearity
def sigmoide(x):
    output = 1/(1+np.exp(-x))
    return  output

# derivada de funcion sigmoide
def sigmoide_derivada(output):
    return output*(1-output)
    
# input dataset
X = np.array([  [0,1],
                [0,1],
                [1,0],
                [1,0] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# inicializos pesos de forma aleatoria con media 0
pesos = 2*np.random.random((2,1)) - 1
historial=[]
for iter in range(10000):

    # forward propagation
    entrada = X
    salida = sigmoide(np.dot(entrada,pesos))

    # how much did we miss?
    error = salida - y
    
#    historial.append(error)

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    salida_delta = error * sigmoide_derivada(salida)
    pesos_derivada = np.dot(entrada.T,salida_delta)

    # actualizamos pesos
    pesos = pesos - pesos_derivada

print("Salida despues de entrenamiento:")
print(salida)
print(" \n Los pesos")
print(pesos, "\n")

#print(salida_error)
#print(historial)

