# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:50:35 2020

@author: ganda

Solution of time independant Schrödinger equation using imaginary time propagation
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
x1d=np.linspace(-5,5,101)
N=len(x1d)
dx=(x1d[N-1]-x1d[0])/(N-1)
x=np.outer(x1d,np.ones(N)) # makes a NxN matrix of x
y=x.T # y is transpose of x
dt=-0.1j # Imaginary time propagation

# Setting up k space
k1d=fftfreq(N,dx/(2*cn.pi))
kx=np.outer(k1d,np.ones(N)) # makes a Nxn matrix of k
ky=kx.T # ky as transpose of kx
kmag=(kx**2+ky**2)**0.5 # magnitude of k

# Guess initial psi as gaussian
psi=0.5*np.exp(-(x**2+y**2))
#psi=np.cos(cn.pi*x/20)*np.cos(cn.pi*y/20)

# Defining potential operator V
w=1 # Angular velocity
g=50 # Contact coefficient
Cdd=0 # Dipole-dipole interaction coefficient

def Vdd(psi): # Dipole interaction energy
    Rc=10 # Circular cut off should be greater than system size
    Uddf=1/3*Cdd*np.nan_to_num((1+3*np.cos(Rc*kmag)/(Rc*kmag)**2-3*np.sin(Rc*kmag)/(Rc*kmag)**3)\
        *(3*(kx/kmag)**2 -1)) # FT of the dipole energy. Assumes polarsation along x
    
    return ifft2(Uddf*fft2(np.abs(psi)**2))

def V(psi): # Harmonic potential
    return 0.5*w**2*(x**2+y**2) + g*np.abs(psi)**2 + Vdd(psi)
"""
def V(psi): # Box potential
    V=np.outer(np.zeros(N),np.ones(N))
    V[0][:]=1e6
    V[N-1][:]=1e6
    for i in range(N): V[i][0]=1e6
    for i in range(N): V[i][N-1]=1e6
    for i in range(N):
        for j in range(N):
            V[i][j]+=g*psi[i][j]**2 + Vdd(psi)[i][j]
    return V
"""
"""
# Circular potential
def V(psi):
    r=3 # Radius of circlar potential
    V=np.outer(np.ones(N)*1e6,np.ones(N))
    binaryMap= x**2 + y**2 <=r**2
    for i in range(N):
        for j in range(N):
            if binaryMap[i][j]==True:
                V[i][j]=g*psi[i][j]**2 + Vdd(psi)[i][j]
    return V
"""

T=0.5*(kx**2+ky**2) # Defining kinetic energy operator T

# Defining operators
expVh=np.exp(-0.5j*V(psi)*dt)
expT=np.exp(-1j*T*dt)

isConv=0;
i=1;

for i in tqdm(range(100)): # Loop until convergence or limit reached
    
    psi=expVh*ifft2(expT*fft2(expVh*psi)) # Split step Fourier method  
    psi/=la.norm(psi) 
    i+=1

# Plotting results
    
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.plot_surface(x,y,psi.real,cmap="jet")

#actual=np.exp(-0.5*(x**2+y**2))
actual=np.cos(cn.pi*x/10)*np.cos(cn.pi*y/10)
actual/=la.norm(actual)
psidiff=psi.real-actual
#ax.plot_surface(x,y,psidiff,cmap="jet")

plt.xlabel("x")
plt.ylabel("y")
ax.set_title("Ground state wavefunction")
plt.show()
    
# Plot heatmap
ax = sns.heatmap(psi.real, xticklabels=x1d, yticklabels=x1d)
ax.set_xticks(ax.get_xticks()[::5])
ax.set_xticklabels(x1d[::5])
ax.set_yticks(ax.get_yticks()[::5])
ax.set_yticklabels(x1d[::5])

