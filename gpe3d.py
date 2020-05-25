# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:50:35 2020

@author: ganda

Modelling a dipolar Bose-Einstein condensate by finding the solution of the 
Gross–Pitaevskii equation using imaginary time propagation
and the split step Fourier method in 3D
"""
import numpy as np
from numpy.fft import fftn,ifftn,fftfreq
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
dx=(x1d[N-1]-x1d[0])/(N)
# Below creates a Nx1x1 column vector of x, 1xNx1 vector of y, and 1x1xN vector of z
x,y,z=x1d.reshape(-1,1,1),x1d.reshape(1,-1,1),x1d.reshape(1,1,-1)
# Below repeats this column vector along the other 2 axes for x,y,z to make a NxNxN matrix
x,y,z=x.repeat(N,1).repeat(N,2),y.repeat(N,0).repeat(N,2),z.repeat(N,0).repeat(N,1)
dt=-0.1j # Imaginary time propagation

# Setting up k space
k1d=fftfreq(N,dx/(2*cn.pi))
# The same process of x,y,z but for k
kx,ky,kz=k1d.reshape(N,1,1),k1d.reshape(1,N,1),k1d.reshape(1,1,N)
kx,ky,kz=kx.repeat(N,1).repeat(N,2),ky.repeat(N,0).repeat(N,2),kz.repeat(N,0).repeat(N,1)
kmag=(kx**2+ky**2+kz**2)**0.5 # magnitude of k

# Guess initial psi as gaussian
psi=0.5*np.exp(-(x**2+y**2+z**2))
#psi=np.cos(cn.pi*x/20)*np.cos(cn.pi*y/20)*np.cos(cn.pi*z/20)

# Defining potential operator V
wx=1 # Angular velocities
wy=1
wz=1
g=0 # Contact coefficient
Cdd=0 # Dipole-dipole interaction coefficient

def Vdd(psi): # Dipole interaction energy
    Rc=5 # Spherical cut off should be greater than system size
    Uddf=1/3*Cdd*np.nan_to_num((1+3*np.cos(Rc*kmag)/(Rc*kmag)**2-3*np.sin(Rc*kmag)/(Rc*kmag)**3)\
        *(3*(kx/kmag)**2 -1)) # FT of the dipole energy. Assumes polarisation along x
    
    return ifftn(Uddf*fftn(np.abs(psi)**2))

def expVh(psi): # Harmonic potential
    V=0.5*((wx*x)**2+(wy*y)**2+(wz*z)**2) + g*np.abs(psi)**2 + Vdd(psi)
    return np.exp(-0.5j*V*dt)

#def expVh(psi): # Box potential
 #   V=((1/4.75)*x)**1000+((1/4.75)*y)**1000+((1/4.75)*z)**1000+g*np.abs(psi)**2 + Vdd(psi)
  #  return np.exp(-0.5j*V*dt)

"""
# Pancake potential
def expVh(psi):
    r=3 # Radius of circlar potential
    V1d=np.ones(N).astype("complex128") * 1e6
    V=V1d.reshape(N,1,1).repeat(N,1).repeat(N,2) # Creates a NxNxN matrix of infinite values 
    binaryMap= x[:,:,0]**2 + y[:,:,0]**2 <=r**2 # All points inside the circle are True
    for i in range(N):
        for j in range(N):
            if binaryMap[i][j]==True: 
                V[i,j,:]=0 # V=0 for all points inside the circle
                
    V+= 0.5*(wz*z)**2 + g*np.abs(psi)**2 + Vdd(psi)
    return np.exp(-0.5j*V*dt)
"""

T=0.5*(kx**2+ky**2+kz**2) # Defining kinetic energy operator T
expT=np.exp(-1j*T*dt)


for i in tqdm(range(20)): # Loop until convergence or limit reached
    psi=expVh(psi)*ifftn(expT*fftn(expVh(psi)*psi)) # Split step Fourier method  
    psi/=la.norm(psi) 
    
    
# Plotting results
zval=0 # Z value on axis
zindex=int(N/2 + zval/dx) # Converts the z value into an index for the z array
fig=plt.figure()
ax=plt.axes(projection="3d")
ax.plot_surface(x[:,:,zindex],y[:,:,zindex],psi[:,:,zindex].real,cmap="jet")

actual=np.exp(-0.5*(wx*x**2+wy*y**2+wz*z**2))
#actual=np.cos(cn.pi*x/10)*np.cos(cn.pi*y/10)*np.cos(cn.pi*z/10)
actual/=la.norm(actual)
psidiff=psi.real-actual
#ax.plot_surface(x[:,:,zindex],y[:,:,zindex],actual[:,:,zindex],cmap="jet")


plt.xlabel("x")
plt.ylabel("y")
ax.set_title("Ground state wavefunction for z="+str(zval))
plt.show()
    
# Plot heatmap
#ax = sns.heatmap(psi[:,:,zindex].real, xticklabels=x1d, yticklabels=x1d)
ax.set_xticks(ax.get_xticks()[::5].astype(int))
ax.set_xticklabels(x1d[::5].astype(int))
ax.set_yticks(ax.get_yticks()[::5].astype(int))
ax.set_yticklabels(x1d[::5].astype(int))

#print("Average error: ",np.abs(np.mean(psidiff)))