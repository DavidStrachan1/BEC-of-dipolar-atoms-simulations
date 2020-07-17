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
from hankel import hankel_class
from cylindercutoff import cyl_cutoff
import pandas as pd
import warnings
warnings.simplefilter('ignore')

# Create grid of size NxN up to max_r in r space and -max_r/2 to max_r/2 in z space
def create_grid(N,max_r,max_z,hankel,hankel_space=False):
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
"""
#function used in U_Cyl
def f_cyl(rho,z,k_rho,k_z):
    return rho*np.cos(k_z*z)*(rho**2-2*z**2)*(rho**2+z**2)**(-5/2)*sp.j0(2*cn.pi*k_rho*rho)

#Cylindrical cut-off integral term
def U_Cyl(kx,ky,kz,R_c,Z_c):
    
    k_rho=np.sqrt(kx**2+ky**2)
    Nx,Ny,Nz=kx.shape[0:3]
    U=np.zeros((Nx,Ny,Nz))
    
    Nri=101
    Nzi=101
    
    dr=np.minimum(0.01*2*cn.pi/k_rho,30/Nri)
    dz=Z_c/(Nzi-1)
      
    for i in tqdm(range(Nri),leave=False):
        for j in range(Nzi):
            U+=dr*dz*f_cyl(R_c+i*dr,j*dz,k_rho,kz)    
    return U
"""
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
    energy = 2*np.pi*dr*dz*np.sum(integrand)  # Doing the integral
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
def graph2D(func,r,z,gamma,D):
    fig=plt.figure()
    ax=plt.axes(projection="3d")
    ax.plot_surface(r,z,func.real,cmap="jet")
    
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
gamma=7 # wz/w trap aspect ratio

#Dimensionless unit formulas
xs=np.sqrt(cn.hbar/(m*w)) # Length scaling parameter
gs=4*cn.pi*a*Na/xs # Dimensionless contact coefficient
D=m*Na*Cdd/(4*cn.pi*(cn.hbar)**2*xs) # Dimensionless dipolar interaction parameter

#Override of parameters:
D=30
gs=0

#Numerical simulation parameters
N=300
max_r=20
max_z=10
dt=-0.01j # Imaginary time propagation unit
Nit=500 # Number of iterations
Rc=5 # r cutoff
Zc=6 # z cutoff

# big dt causing nan? better initial guess

## TODO
# max_z/rootgamma
# b=max_z/2
# xs as a function of wz for box plot

# Define grids
hankel=hankel_class(N,max_r)
r,rho = create_grid(N,max_r,max_z,hankel,True)
z,kz = create_grid(N,max_r,max_z,hankel,False)

k_rho=2*np.pi*rho # rho is wavevector in hankel space
J=hankel.J.reshape(-1,1) # Used in hankel transform

# Magnitude of k^2
kmag=(k_rho**2+kz**2)**0.5 

# Harmonic potential
V=V_harmonic(r,z,1,gamma)

# Kinetic energy operator
T=0.5*kmag**2
expT=np.exp(-1j*T*dt)

# Uddf calculation
DD=D*4*np.pi/3
Uddf=3*np.nan_to_num(kz/kmag,posinf=0)**2-1

if gamma<5: # Use spherical cutoff
    Uddf*=1+3*np.nan_to_num(np.cos(Rc*kmag)/(Rc*kmag)**2,posinf=0) \
        -3*np.nan_to_num(np.sin(Rc*kmag)/(Rc*kmag)**3,posinf=0)
else:
    Uddf+=np.exp(-Zc*k_rho)*(np.nan_to_num(k_rho/kmag,posinf=0)**2*np.cos(Zc*kz) \
                             -np.nan_to_num(k_rho/kmag,posinf=0)*np.nan_to_num(kz/kmag,posinf=0)\
                                 *np.sin(Zc*kz))
    #Uddf-=U # R cutoff integral (if=0 then just z cutoff)
Uddf*=DD

# Guess initial psir as gaussian
psi=np.exp(-r**2/5-gamma*z**2)
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
    energies.append(E(psi,N,max_r,max_z,V,Uddf,gs,Na,kmag,hankel,J))
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

# Plotting results
graph2D(psi,r,z,gamma,D)

# Linear plots
plt.plot(r[:,0],psi[:,int(N/2)]);
#plt.plot(z,psi[0,:])

# Plot heatmap
#sns.heatmap(psi.real)

#print("Average error: ",np.abs(np.mean(psidiff)))
