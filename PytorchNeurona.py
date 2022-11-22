#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 21:25:50 2022

@author: jorge
"""
import numpy as np
import torch
N=100
a=0
b=20
y_euler=torch.ones(N)
y_euler[0]=0
x=torch.Tensor(np.linspace(a,b,N)[:,None])
h=(b-a)/N
for i in range(N-1):
    y_euler[i+1]=y_euler[i] + h*((-1/5)*y_euler[i] + torch.exp(-x[i]/5)*torch.cos(x[i]))

y=torch.exp(-(x/5))*torch.sin(x)   
import matplotlib.pyplot as plt
fig, ax= plt.subplots()
ax.plot(x.data.numpy(), y_euler.data.numpy(), "orange", label="Solucion por red Neuronal")
ax.plot(x.data.numpy(), y.data.numpy(), "g--", label="Solucion  Analitica")
plt.legend()