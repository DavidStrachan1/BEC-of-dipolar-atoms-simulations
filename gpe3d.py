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
from cylindercutoff import cyl_cutoff
import warnings
warnings.simplefilter('ignore')

# Defining size of x space
x1d=np.linspace(-5,5,100)
N=len(x1d)
dx=(x1d[N-1]-x1d[0])/(N)
# Below creates a Nx1x1 column vector of x, 1xNx1 vector of y, and 1x1xN vector of z
x,y,z=x1d.reshape(N,1,1),x1d.reshape(1,N,1),x1d.reshape(1,1,N)
# Below repeats this column vector along the other 2 axes for x,y,z to make a NxNxN matrix
x,y,z=x.repeat(N,1).repeat(N,2),y.repeat(N,0).repeat(N,2),z.repeat(N,0).repeat(N,1)
dt=-0.01j # Imaginary time propagation

# Setting up k space
k1d=fftfreq(N,dx/(2*cn.pi))
# The same process of x,y,z but for k
kx,ky,kz=k1d.reshape(N,1,1),k1d.reshape(1,N,1),k1d.reshape(1,1,N)
kx,ky,kz=kx.repeat(N,1).repeat(N,2),ky.repeat(N,0).repeat(N,2),kz.repeat(N,0).repeat(N,1)
kmag=(kx**2+ky**2+kz**2)**0.5 # magnitude of k

# Units and dimensional stuff
m=2.7774e-25 # Typical mass of erbium
a=5.1e-9 # Typical s-wave scattering length
w=20*cn.pi # Typical r angular velocity
gamma=8 # wz/w
xs=np.sqrt(cn.hbar/(m*w)) # Length scaling parameter
n=1e5 # Typical number of particles
gs=4*cn.pi*a*n/xs # Dimensionless contact coefficient
a0=5.29e-11 # Bohr radius
bohr_mag=cn.e*cn.hbar/(2*cn.m_e) # Bohr magneton
Cdd=cn.mu_0*(6.98*bohr_mag)**2 # Dipole-dipole interaction coefficient
D=m*n*Cdd/(4*cn.pi*(cn.hbar)**2*xs) # Dimensionless dipolar interaction parameter

D=3000
gs=0

# Potential energy operator
def V(psi):
    return 0.5*(x**2+y**2+(gamma*z)**2) + gs*np.abs(psi)**2 + Vdd_cyl(psi)

# Returns dimensionless energy of wavefunction
def E(psi):
    integrand=0.5*np.abs(np.gradient(psi))**2 + V(psi)*np.abs(psi)**2 \
        + (gs/2)*np.abs(psi)**4 + (D/2)*Vdd(psi)*np.abs(psi)**2
            
    integral =  dx**3*np.sum(integrand)
    
    return integral

            
def Vdd(psi): # Dipole interaction energy
    Rc=5 # Spherical cut off should be greater than system size
    Uddf=D/3
    Uddf+=D*np.cos(Rc*kmag)/(Rc*kmag)**2
    Uddf=np.nan_to_num(Uddf)
    Uddf-=D*np.sin(Rc*kmag)/(Rc*kmag)**3
    Uddf=np.nan_to_num(Uddf)
    Uddf*= np.nan_to_num(3*(kz/kmag)**2 -1) # FT of the dipole energy. Assumes polarisation along z
    return ifftn(Uddf*fftn(np.abs(psi)**2))

def Vdd_cyl(psi): # Dipole interaction energy with cylindrical cut-off
    Zc=np.max(z)+1 # z cut-off
    k_rho=np.sqrt(kx**2+ky**2)
    theta_k=np.arccos(kz/kmag) # Dipoles polarized in z direction
    # Calculating FT of dipole potential
    Uddf=1/3*(3*(kz/kmag)**2-1)
    Uddf=np.nan_to_num(Uddf)
    Uddf+=np.exp(-Zc*k_rho)\
        *((np.sin(theta_k))**2*np.cos(Zc*kz)-np.sin(theta_k)*(kz/kmag)*np.sin(Zc*kz))
    Uddf=np.nan_to_num(Uddf)
    Uddf-= cyl_cutoff.U
    Uddf*=D
    return ifftn(Uddf*fftn(np.abs(psi)**2)) # Convolution theorem

def expVh(psi): # Harmonic potential
    #V=0.5*(x**2+y**2+(gamma*z)**2) + gs*np.abs(psi)**2 + Vdd(psi)
    return np.exp(-0.5j*V(psi)*dt)

#def expVh(psi): # Box potential
 #   V=((1/4.75)*x)**1000+((1/4.75)*y)**1000+((1/4.75)*z)**1000+gs*np.abs(psi)**2 + Vdd(psi)
  #  return np.exp(-0.5j*V*dt)

"""
# Pancake potential
def expVh(psi):
    r=3 # Radius of circlar potential
    V1d=np.ones(N).astype("complex128") * 1e6
    V=V1d.reshape(N,1,1).repeat(N,1).repeat(N,2) # Creates a NxNxN matrix of infinite values 
    binaryMap= (x[:,:,0]**2 + y[:,:,0]**2 >=r**2).astype(complex)*1e6 # All points inside the circle are 0
                
    V+= 0.5*(wz*z)**2 + g*np.abs(psi)**2 + Vdd(psi)
    return np.exp(-0.5j*V*dt)
"""

T=0.5*(kx**2+ky**2+kz**2) # Defining kinetic energy operator T
expT=np.exp(-1j*T*dt)

# Guess initial psi as gaussian
psi=0.5*np.exp(-(x**2+y**2+z**2))
#psi=np.cos(cn.pi*x/20)*np.cos(cn.pi*y/20)*np.cos(cn.pi*z/20)

dt_array=np.linspace(-0.1j,0.1j/gamma,500)

for i in tqdm(range(500)): # Loop until limit reached
    #dt=dt_array[i]
    psi=expVh(psi)*ifftn(expT*fftn(expVh(psi)*psi)) # Split step Fourier method  
    psi/=la.norm(psi) 
       
actual=np.exp(-0.5*(x**2+y**2+z**2))
#actual=np.cos(cn.pi*x/10)*np.cos(cn.pi*y/10)*np.cos(cn.pi*z/10)
actual/=la.norm(actual)
grpsidiff=psi.real-actual    
    
zval=0 # Z value on axis    
# Plotting results
def graph(psi,zval):

    zindex=int(N/2 + zval/dx) # Converts the z value into an index for the z array
    fig=plt.figure()
    ax=plt.axes(projection="3d")
    ax.plot_surface(x[:,:,zindex],y[:,:,zindex],psi[:,:,zindex].real,cmap="jet")
    
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_title("Ground state wavefunction for z="+str(zval))
    plt.show()
    
graph(psi,zval)   

"""
# Plot heatmap
#ax = sns.heatmap(psi[:,:,zindex].real, xticklabels=x1d, yticklabels=x1d)
ax.set_xticks(ax.get_xticks()[::5].astype(int))
ax.set_xticklabels(x1d[::5].astype(int))
ax.set_yticks(ax.get_yticks()[::5].astype(int))
ax.set_yticklabels(x1d[::5].astype(int))
"""
#print("Average error: ",np.abs(np.mean(psidiff)))
