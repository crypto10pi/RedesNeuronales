#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:12:17 2022

@author: jorge
"""
import numpy as np
import torch

def sigmoide(x):
    return 1/(1+torch.exp(-x))

def sigmoide_grad(x):
    return sigmoide(x)*(1-sigmoide(x))

def red_neuronal(x, weights, bias):
    s_z =sigmoide(torch.matmul(x,weights[0])+ bias[0])
    return torch.matmul(s_z, weights[1]) + bias[1] 

def dN_dx(weights,x):
    s_z_grad = sigmoide_grad(torch.matmul(x,weights[0]) + bias[0])
    mul=torch.mul(weights[0].T, weights[1])
    return torch.matmul(s_z_grad, mul)

weights = [torch.randn((1,10), requires_grad=True), torch.randn((10,1), requires_grad=True)]   
bias = [torch.randn(10, requires_grad=True), torch.randn(1, requires_grad=True)]       
A=0
Psi_t =lambda x: A + x*red_neuronal(x, weights, bias)
f = lambda x , Psi: torch.exp(-x/5.0)*torch.cos(x) - Psi/ 5.0

def error(x):
    x.requires_grad = True
    psi= Psi_t(x)
    ddN = dN_dx(weights, x)
    Psi_t_x =red_neuronal(x, weights,bias) + x*ddN
    return torch.mean( ( Psi_t_x - f(x, psi)) **2)

epochs = 10000
lr=0.01
N=100
a=0
b=5
x=torch.unsqueeze(torch.linspace(a,b,N), dim=1)  

for i in range(epochs):
    loss= error(x)
    loss.backward()
    weights[0].data -= lr*weights[0].grad.data
    weights[1].data -= lr*weights[1].grad.data
    bias[0].data -= lr*bias[0].grad.data
    bias[1].data -= lr*bias[1].grad.data
    
    weights[0].grad.zero_()
    weights[1].grad.zero_()
    bias[0].grad.zero_()
    bias[1].grad.zero_()
    
    print("Loss", loss.item())
    
 
y=torch.exp(-(x/5))*torch.sin(x) 
psi_trial = Psi_t(x) 


import matplotlib.pyplot as plt   
fig, ax= plt.subplots()
ax.plot(x.data.numpy(), psi_trial.data.numpy(), "orange", label="Solucion por red Neuronal")
ax.plot(x.data.numpy(), y.data.numpy(), "g--", label="Solucion  Analitica")
plt.legend() 
 