# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:50:35 2020

@author: ganda

Solution of time independant Schrödinger equation using imaginary time propagation
and the split step Fourier method in 1D
"""
import numpy as np
from numpy.fft import fft,ifft,fftfreq
from numpy import linalg as la
import matplotlib.pyplot as plt
import scipy.constants as cn
import warnings
warnings.simplefilter('ignore')

# Defining size of x space
x=np.linspace(-5,5,101)
N=len(x)
dx=(x[N-1]-x[0])/(N-1)
dt=-0.1j # Imaginary time propagation

# Setting up k space
k=fftfreq(N,dx/(2*cn.pi)) 

# Guess initial psi as gaussian
psi=0.5*np.exp(-x**2)
#psi=np.cos(cn.pi*x/20)


# Defining potential V and kinetic energy T
w=1 # Angular velocity
#V=0.5*w**2*x**2
#V=np.zeros(N)
V=x**(20)
#V[0]=1e6
#V[N-1]=1e6
T=0.5*k**2

# Defining operators
expVh=np.exp(-0.5j*V*dt)
expV=np.exp(-1j*V*dt)
expT=np.exp(-1j*T*dt)

# Create historical psi array
histpsi=[[0,psi[0],psi[int(round(N/2))],psi[N-1]]]

isConv=0;
i=1;

while isConv==0 and i<1000: # Loop until convergence or limit reached
    
    psi=expVh*ifft(expT*fft(expVh*psi)) # Split step Fourier method
    psi/=la.norm(psi)
     
    # Add psi to the history of psi
    histpsi.append([i,psi[0],psi[int(round(N/2))],psi[N-1]])
    
    # Checking for convergence
    if abs(histpsi[i][1]-histpsi[i-1][1])<1e-6 and abs(histpsi[i][2]-histpsi[i-1][2])<1e-6\
        and abs(histpsi[i][3]-histpsi[i-1][3])<1e-6:
        isConv=1
        
    i+=1
  
# Storing the index and value of histpsi into arrays to be able to plot    
xhist=[i[0] for i in histpsi]
yhist=[i[2] for i in histpsi]

# Plotting results
    
#actual=np.exp(-0.5*x**2)
actual=np.cos(cn.pi*x/2)
actual/=la.norm(actual)
diff=psi-actual
plt.plot(x,psi,color="k",label="Numerical result")
#plt.plot(x,actual,color="g",label="Actual result") 
#plt.plot(x,diff,color="orange",label="Difference between numerical and actual results")  
plt.xlabel("Distance")
plt.ylabel("Wavefunction")
plt.legend(loc="upper right")

""" Eigenvalue stuff

H=np.array([[5,1],
            [4,4]])


# Eigenvalues of H
e=[0];

e[0]=np.linalg.eig(H)[0][0]
e.append(np.linalg.eig(H)[0][1])

realE=min(e)

for i in range(1,10):
    psi=psi-dt*H@psi
    psi=psi/np.linalg.norm(psi)
    E=psi@H@np.transpose(psi)
    
"""
    