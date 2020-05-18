# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:50:35 2020

@author: ganda

Solution of time independant Schrödinger equation using imaginary time propagation
and the split step Fourier method
"""
import numpy as np
from numpy.fft import fft,ifft,fftfreq
from numpy import linalg as la
import matplotlib.pyplot as plt
#import scipy.constants as cn
import warnings
warnings.simplefilter('ignore')

# Defining size of x space
x=np.linspace(-5,5,100).astype(np.complex128)
N=len(x)
dx=(x[N-1]-x[0])/N
dt=-0.1j # Imaginary time propagation

# Setting up k space
k=fftfreq(N,dx)      

# Defining potential V and kinetic energy T
w=1 # Angular velocity
V=0.5*w**2*x**2
T=0.5*k**2


# Defining operators
expVh=np.exp(-0.5j*V*dt)
expV=np.exp(-1j*V*dt)
expT=np.exp(-1j*T*dt)


# Defining operators diagonally
#expVh=np.diag(np.exp(-0.5j*V*dt))
#expT=np.diag(np.exp(-1j*T*dt))

# Guess initial psi as gaussian
psi=0.5*np.exp(-x**2)

# Create historical psi array
histpsi=[[1,psi[0],psi[int(round(N/2))],psi[N-1]]]

isConv=0;
i=1;

while isConv==0 and i<1000: # Loop until convergence or limit reached
    
    psi=expVh*ifft(expT*fft(expVh*psi)) # Split step Fourier method
    
    #psi=expVh@ifft(expT@fft(expVh@psi)) # Split step Fourier method diagonal matrix
    
    psi=psi/la.norm(psi)
     
    # Add psi to the history of psi
    histpsi.append([i,psi[0],psi[int(round(N/2))],psi[N-1]])
    
    # Checking for convergence
    if abs(histpsi[i][1]-histpsi[i-1][1])<1e-6 and abs(histpsi[i][2]-histpsi[i-1][2])<1e-6\
        and abs(histpsi[i][3]-histpsi[i-1][3])<1e-6:
        isConv=1
        
    i+=1
  
# Storing the index and value of histpsi into arrays to be able to plot
xhist=[]
yhist=[]
for i in range(1,len(histpsi)):
    xhist.append(histpsi[i][0])
    yhist.append(histpsi[i][2]) #[2] is the middle of the wavefunction

actual=np.exp(-0.5*x**2)
actual/=la.norm(actual)
plt.plot(x,psi,color="k")
#plt.plot(x,actual,color="g")   
plt.xlabel("Distance (m)")
plt.ylabel("Wavefunction")

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
    
