# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:19:30 2020
Modelling a dipolar Bose-Einstein condensate by finding the ground state solution 
of the Gross–Pitaevskii equation using imaginary time propagation
and the split step Hankel method in 3D with cylindrical symmetry
"""
import numpy as np
from numpy.fft import fft,ifft,fftfreq
from numpy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from tqdm import tqdm
import scipy.constants as cn
import scipy.special as sp
from hankel import hankel_class
import pandas as pd
import warnings
warnings.simplefilter('ignore')

# Create grid of size NxN up to max_r in r space and -max_r/2 to max_r/2 in z space
def create_grid(N,max_r,max_z,gamma,hankel,hankel_space=False):
    if hankel_space:
        x=hankel.r().reshape(-1,1).repeat(N+1,1) # Makes r space
        y=hankel.rho().reshape(-1,1).repeat(N+1,1) # Makes rho space
    else:
        dz=2*max_z/N
        x=np.linspace(-max_z,max_z,N+1).reshape(1,-1).repeat(N+1,0) # Makes z space
        y=fftfreq(N+1,dz/(2*np.pi)).reshape(1,-1).repeat(N+1,0) # Makes kz space
    return x,y

# Harmonic potential function (dimensionless)
def V_harmonic(r,z,gamma_r,gamma_z):
    return 0.5*((gamma_r*r)**2+(gamma_z*z)**2)

def V_box(r,z,box_size,gamma_z):
    return (r/(box_size-0.2))**60 + 0.5*(gamma_z*z)**2

# Dipole potential contribution for a given Uddf (FT of dipole term)
def Vdd(psi,Uddf,hankel,J):
    out = Uddf*hankel.hankel(fft(np.abs(psi)**2,axis=1),J)
    return  ifft(hankel.invhankel(out,J),axis=1)

# Returns dimensionless energy of wavefunction
def E(psi,N,max_r,max_z,V_trap,Uddf,gs,Na,kmag,hankel,J):
    dr=max_r/N
    dz=2*max_z/N
    integrand= 0.5*np.abs(ifft(hankel.invhankel(-1j*kmag*fft(hankel.hankel(psi,J),axis=1),J),axis=1))**2\
        + V_trap*np.abs(psi)**2 \
        + (gs/2)*np.abs(psi)**4 + (1/2)*Vdd(psi,Uddf,hankel,J)*np.abs(psi)**2           
    #energy = 2*np.pi*dr*dz*np.sum(integrand)  # Doing the integral
    energy=2*np.pi*np.sum(integrand[1,:])*dz*r[1,0]*r[1,0]/2 # First r term has slightly different dr
    energy+=2*np.pi*np.sum(r[2:,:]*integrand[2:,:])*dr*dz
    return energy.real

# Exp of potential energy operator
def expVh(psi,V_trap,gs,Uddf,dt,J): # spatial term of hamiltonian
    V=V_trap + gs*np.abs(psi)**2/(2*np.pi) + Vdd(psi,Uddf,hankel,J)
    return np.exp(-0.5j*V*dt)

def norm(psi,r,z,N,max_z): # Normalisation function
    dz=2*max_z/N
    out = 2*cn.pi*np.sum(np.abs(r[1,:]*psi[1,:])**2)*dz # First r term has slightly different dr
    out += 2*cn.pi*np.sum(r[2:,:]*np.abs(psi[2:,:])**2)*(r[N-1,0]-r[1,0])*dz/(N-1) # Rest of normalisation using average dr
    out = np.sqrt(out)
    return out

# Plotting results
def graph2D(func,r,z,N,gamma,D):
    fig=plt.figure()
    ax=plt.axes(projection="3d")
    #ax.plot_surface(r,z,func.real,cmap="jet")
    
    midway_index_z = int(N/2)
    z_cutoff = z.max()
    r_cutoff = 10
    bools =(np.abs(r) <= r_cutoff)*(np.abs(z) <=z_cutoff)
    n_1 = np.sum(bools[:,midway_index_z]) # number of r coords with value less than r_cutoff
    n_2 = np.sum(bools[0,:]) # number of z coords with value less than z_cutoff
    z_coord_i = int(midway_index_z-n_2/2)
    z_coord_f = int(midway_index_z+n_2/2)
    ax.plot_surface(r[0:n_1,z_coord_i:z_coord_f],z[0:n_1,z_coord_i:z_coord_f],np.abs(func)[0:n_1,z_coord_i:z_coord_f],cmap='jet')
    
    plt.xlabel("r")
    plt.ylabel("z")
    ax.set_title("Ground state wavefunction for D=%d and γ=%d" %(D,gamma))
    plt.show()
    
# Constants
m=2.7774e-25 # Mass of erbium
a0=5.29e-11 # Bohr radius
bohr_mag=cn.e*cn.hbar/(2*cn.m_e) # Bohr magneton
Cdd=cn.mu_0*(6.98*bohr_mag)**2 # Dipole-dipole interaction coefficient of erbium

#Physical parameters
a=a0*100 # s-wave scattering length
Na=1e5 # Number of atoms
w=20*cn.pi # Radial angular velocity
gamma=14 # wz/w trap aspect ratio

#Dimensionless unit formulas
xs=np.sqrt(cn.hbar/(m*w)) # Length scaling parameter
gs=4*cn.pi*a*Na/xs # Dimensionless contact coefficient
D=m*Na*Cdd/(4*cn.pi*(cn.hbar)**2*xs) # Dimensionless dipolar interaction parameter

#Override of parameters:
D=5
gs=0

#Numerical simulation parameters
N=100
max_z_init=10
dt=-0.01j # Imaginary time propagation unit
Nit=500 # Number of iterations
Rc=8 # radius of r cutoff (Not needed in r direction)
#Zc=6 # z cutoff

#max_r=10 # For harmonic

box_size=3 # Chosen when the gaussian in z goes to 0, so this box size produces a roughly spherical cloud
max_r=1.2*box_size
max_z=max_z_init*gamma**(-0.5) # Changes max_z as a function of gamma (for higher accuracy)
Zc=max_z/2

L=2e-5/box_size

mu=2*Na*Cdd/(3*cn.hbar*cn.pi*L**2*w*xs)
mu_d=Na*Cdd/(3*cn.hbar*cn.pi*L**2*w*xs)

# Define grids
hankel=hankel_class(N,max_r) # Create instance of hankel transform with set parameters
r,rho = create_grid(N,max_r,max_z,gamma,hankel,True)
z,kz = create_grid(N,max_r,max_z,gamma,hankel,False)

k_rho=2*np.pi*rho # rho is wavevector in hankel space
J=hankel.J.reshape(-1,1) # Used in hankel transform

# Magnitude of k^2
kmag=(k_rho**2+kz**2)**0.5 

# Harmonic potential
#V=V_harmonic(r,z,1,gamma)
V=V_box(r,z,box_size,gamma)

# Kinetic energy operator and the exponential form
T=0.5*kmag**2
expT=np.exp(-1j*T*dt)

# Uddf calculation
DD=D*4*np.pi/3
# Use spherical cutoff
Uddf1=(3*np.nan_to_num(kz/kmag,posinf=0)**2-1)\
    *(1+3*np.nan_to_num(np.cos(Rc*kmag)/(Rc*kmag)**2,posinf=0)\
    -3*np.nan_to_num(np.sin(Rc*kmag)/(Rc*kmag)**3,posinf=0))
# Use cylindrical cutoff
Uddf2=(3*np.nan_to_num(kz/kmag,posinf=0)**2-1)+3*np.exp(-Zc*k_rho)*(np.nan_to_num(k_rho/kmag,posinf=0)**2*np.cos(Zc*kz) \
    -np.nan_to_num(k_rho/kmag,posinf=0)*np.nan_to_num(kz/kmag,posinf=0)*np.sin(Zc*kz))
    #Uddf-=U # R cutoff integral (if=0 then just z cutoff)

if gamma<4:
    Uddf=Uddf1 # Use spherical cut-off for low gamma
else:
    Uddf=Uddf2 # Use cylindrical cut-off for high gamma
    
Uddf*=DD

# Guess initial psir as gaussian
#psi=np.exp(-(r**2/5+gamma*z**2))
psi=np.exp(-r**2/50-gamma*z**2/50)
psi/=norm(psi,r,z,N,max_z)

# Imaginary time propagation
energies=[]
hist_psi=[]
index=range(Nit)

for i in tqdm(index,leave=False): # Loop until limit reached
    psi=expVh(psi,V,gs,Uddf,dt,J)*psi # Split step Fourier/Hankel method
    psi=expT*fft(hankel.hankel(psi,J),axis=1)
    psi/=norm(psi,r,z,N,max_z)
    psi=ifft(hankel.invhankel(psi,J),axis=1)
    psi/=norm(psi,r,z,N,max_z)
    psi*=expVh(psi,V,gs,Uddf,dt,J)
    psi/=norm(psi,r,z,N,max_z)
    
    # if i%5==0: # Runs every 5 iterations
    # Add energy of psi to an existing list
    energies.append(E(psi,N,max_r,max_z,V,Uddf,gs,Na,kmag,hankel,J))
    # Add sum of wavefunction to an existing list
    hist_psi.append(np.sum(psi))
    
energies=np.array(energies)
hist_psi=np.array(hist_psi)

## Thomas-Fermi calculations
w_bar=gamma**(1/3)*w
a_bar=np.sqrt(cn.hbar/(m*w_bar))

mu=0.5*15**(2/5)*(Na*a/a_bar)**(2/5)*cn.hbar*w_bar # Chemical potential
tf_radius=np.sqrt(2*mu/(m*w**2))/xs # Thomas-Fermi radius

tf=np.nan_to_num(np.sqrt((mu/(cn.hbar*w)-V)/gs))
tf/=norm(tf,r,z,N,max_z)
tf_diff=tf-psi

## Actual psi
actual=np.exp(-0.5*(r**2+gamma*z**2))
#actual=np.cos(r/cn.pi)*np.exp(-0.5*gamma*z**2)*(r<max_r/2).astype(int) # Not correct ground state
actual=sp.j0(sp.jn_zeros(0,1)[0]*r/(box_size))*(r<box_size).astype(int)*np.exp(-0.5*gamma*z**2)
actual/=norm(actual,r,z,N,max_z)
psidiff=psi.real-actual

# Plotting results
graph2D(psi,r,z,N,gamma,D)

# Linear plots
plt.plot(r[:,0],psi[:,int(N/2)]);
#plt.plot(z[0,:],psi[0,:]);

# Plot heatmap
#sns.heatmap(psi.real)

#print("Average error: ",np.abs(np.mean(psidiff)))
