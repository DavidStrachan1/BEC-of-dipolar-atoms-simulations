# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:50:35 2020

@author: ganda

Modelling a dipolar Bose-Einstein condensate by finding the solution of the 
time dependant Schrödinger equation using imaginary time propagation
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
x=np.linspace(-10,10,1000)
N=len(x)
dx=(x[N-1]-x[0])/N
# Imaginary time propagation (dt<<ts for convergence)
# ts=1/w for harmonic potential (for w= 20π, ts=0.016) 
#and ts=mL^2/hbar for a box (For L=20, ts=0.8)
dt=-0.005j

# Setting up k space
k=fftfreq(N,dx/(2*cn.pi)) 

# Guess initial psi as gaussian
psi=0.5*np.exp(-x**2)
#psi=np.ones(N)
#psi=x**2
#psi=np.cos(cn.pi*x/20)

# Units and dimensional stuff
m=2.7774e-25 # Typical mass of erbium
a=5.1e-9 # Typical s-wave scattering length
w=20*cn.pi # Typical angular velocity
xs=np.sqrt(cn.hbar/(m*w)) # Length scaling parameter
n=1e5 # Typical number of particles
gs=4*cn.pi*a*n/xs # Dimensionless contact coefficient
bohr_mag=cn.e*cn.hbar/(2*cn.m_e) # Bohr magneton
#Cdd=cn.mu_0*(6.98*bohr_mag)**2/(4*cn.pi) # Dipole-dipole interaction coefficient
Cdd=0

def Vdd(psi): # Dipole interaction energy
    Rc=10 # Circular cut off should be greater than system size
    Uddf=1/3*Cdd*np.nan_to_num((1+3*np.cos(Rc*k)/(Rc*k)**2-3*np.sin(Rc*k)/(Rc*k)**3)\
        *(3*(0)**2 -1)) # FT of the dipole energy. Assumes polarsation along x
    return ifft(Uddf*fft(np.abs(psi)**2))     
    

def expVh(psi): # Harmonic potential
    V=0.5*x**2 + gs*np.abs(psi)**2 + Vdd(psi)
    return np.exp(-0.5j*V*dt)
  

#def expVh(psi): # Box potential
 #   V=(x/9.9)**1000 + gs*np.abs(psi)**2 + Vdd(psi)
  #  return np.exp(-0.5j*V*dt)

# Defining kinetic energy operator T
T=0.5*k**2

# Defining operators
expT=np.exp(-1j*T*dt)

# Create historical psi array
histpsi=[[0,psi[0],psi[int(round(N/2))],psi[N-1]]]

isConv=0;
i=1;

while isConv==0 and i<1000: # Loop until convergence or limit reached
    
    psi=expVh(psi)*ifft(expT*fft(expVh(psi)*psi)) # Split step Fourier method
    psi/=la.norm(psi)
     
    # Add psi to the history of psi
    histpsi.append([i,psi[0],psi[int(round(N/2))],psi[N-1]])
    
    # Checking for convergence
    if abs(histpsi[i][1]-histpsi[i-1][1])<1e-6 and abs(histpsi[i][2]-histpsi[i-1][2])<1e-6\
        and abs(histpsi[i][3]-histpsi[i-1][3])<1e-6:
        isConv=0
        
    i+=1
  
# Storing the index and value of histpsi into arrays to be able to plot    
xhist=[i[0] for i in histpsi]
yhist=[i[2] for i in histpsi]

# Plotting results
    
actual=np.exp(-0.5*x**2)
#actual=np.cos(cn.pi*x/20)
actual/=la.norm(actual)
diff=psi-actual
plt.plot(x,psi,color="k",label="Numerical result")
plt.plot(x,actual,color="g",label="Actual result") 
#plt.plot(x,diff,color="orange",label="Difference between numerical and actual results")  
plt.xlabel("Distance (μm)")
plt.ylabel("Wavefunction")
plt.legend(loc="upper right")
