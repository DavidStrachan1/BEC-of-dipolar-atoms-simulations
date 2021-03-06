# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:50:35 2020

@author: ganda

Modelling a dipolar Bose-Einstein condensate by finding the solution of the 
time dependant Schrödinger equation using imaginary time propagation
and the split step Fourier method in 2D
"""
import numpy as np
from numpy.fft import fft2,ifft2,fftfreq
from numpy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from tqdm import tqdm
import scipy.constants as cn
import warnings
warnings.simplefilter('ignore')


# Defining size of x space
x1d=np.linspace(-5,5,100)
N=len(x1d)
dx=(x1d[N-1]-x1d[0])/N
xv,yv=np.meshgrid(x1d,x1d) # makes a NxN matrix of x
dt=-0.01j # Imaginary time propagation

# Setting up k space
k1d=fftfreq(N,dx/(2*cn.pi))
kx,ky=np.meshgrid(k1d,k1d) # makes a NxN matrix of k
kmag=(kx**2+ky**2)**0.5 # magnitude of k

# Guess initial psi as gaussian
psi=0.5*np.exp(-(xv**2+yv**2))
#psi=np.cos(cn.pi*xv/20)*np.cos(cn.pi*yv/20)
#psi=np.ones((N,N))

# Defining potential operator V
wx=1 # Angular velocities
wy=1
g=0 # Contact coefficient
Cdd=0 # Dipole-dipole interaction coefficient

def Vdd(psi): # Dipole interaction energy
    Rc=5 # Circular cut off should be greater than system size
    Uddf=1/3*Cdd*np.nan_to_num((1+3*np.cos(Rc*kmag)/(Rc*kmag)**2-3*np.sin(Rc*kmag)/(Rc*kmag)**3)\
        *(3*(kx/kmag)**2 -1)) # FT of the dipole energy. Assumes polarisation along x
    
    return ifft2(Uddf*fft2(np.abs(psi)**2))

#def expVh(psi): # Harmonic potential
 #   V=0.5*((wx*xv)**2+(wy*yv)**2) + g*np.abs(psi)**2 + Vdd(psi)
  #  return np.exp(-0.5j*V*dt)

#def expVh(psi): # Box potential
 #   V=((1/4.75)*xv)**1000 + ((1/4.75)*yv)**1000 + g*np.abs(psi)**2 + Vdd(psi)
  #  return np.exp(-0.5j*V*dt)

"""
# Circular potential
def expVh(psi):
    r=3 # Radius of circlar potential
    V=np.outer(np.ones(N)*1e6,np.ones(N)).astype("complex128")
    binaryMap= xv**2 + yv**2 <=r**2 # All points inside the circle are True
    for i in range(N):
        for j in range(N):
            if binaryMap[i][j]==True: # V=0 for all points inside the circle
                V[i][j]=0
                
    V+=g*np.abs(psi)**2 + Vdd(psi)
    return np.exp(-0.5j*V*dt)
"""

T=0.5*(kx**2+ky**2) # Defining kinetic energy operator T
expT=np.exp(-1j*T*dt)

for i in tqdm(range(100)): # Loop until convergence or limit reached
    
    psi=expVh(psi)*ifft2(expT*fft2(expVh(psi)*psi)) # Split step Fourier method  
    psi/=la.norm(psi) 
    
# Plotting results  
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.plot_surface(xv,yv,psi.real,cmap="jet")

actual=np.exp(-0.5*(wx*xv**2+wy*yv**2))
#actual=np.cos(cn.pi*xv/10)*np.cos(cn.pi*yv/10)
actual/=la.norm(actual)
psidiff=psi.real-actual
#ax.plot_surface(xv,yv,psidiff,cmap="jet")


plt.xlabel("x")
plt.ylabel("y")
ax.set_title("Ground state wavefunction")
plt.show()
    
# Plot heatmap
#ax = sns.heatmap(psi.real, xticklabels=x1d, yticklabels=x1d)
ax.set_xticks(ax.get_xticks()[::5].astype(int))
ax.set_xticklabels(x1d[::5].astype(int))
ax.set_yticks(ax.get_yticks()[::5].astype(int))
ax.set_yticklabels(x1d[::5].astype(int))

#print("Average error: ",np.abs(np.mean(psidiff)))
