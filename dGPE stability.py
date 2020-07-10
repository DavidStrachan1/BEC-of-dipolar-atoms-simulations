# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 22:29:46 2020

@author: ganda


Testing for stability of dGPE
"""
import numpy as np
import math
from numpy.fft import fft,ifft,fftfreq,fft2,ifft2
from numpy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tqdm import tqdm
import scipy.constants as cn
from hankel import hankel_class as hankel
from cylindercutoff import cyl_cutoff
import warnings
warnings.simplefilter('ignore')

z=np.linspace(-10,10,301) # Size of z space 
N=len(z)
dz=(z[N-1]-z[0])/(N)
dt=-0.01j # Imaginary time propagation

# r and rho space
r=hankel.r()
rho1d=hankel.rho()
J=hankel.J()

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

D=400
gs=0

### Below is code for finding the ground state

#rv,zv=np.meshgrid(r,z) # Make a 2D grid of r and z 
rv=r.reshape(1,-1).repeat(N,0).T
zv=z.reshape(-1,1).repeat(N,1).T

rho=rho1d.reshape(1,-1).repeat(N,0).T # Turning rho into a NxN matrix

# Setting up k space for the z axis
k1d=fftfreq(N,dz/(2*np.pi))
kz=k1d.reshape(-1,1).repeat(N,1).T
kmag=np.sqrt(kz**2+(2*np.pi*rho)**2).T

# Kinetic energy operator along r (hankel tranform diagonalises T in r space)
Tr=0.5*(2*np.pi*rho1d)**2
expTr=np.exp(-1j*Tr*dt)
expTr=expTr.reshape(-1,1)

# Kinetic energy operator along z axis
Tz=0.5*kz**2
expTz=np.exp(-1j*Tz*dt)

# Returns dimensionless energy of wavefunction
def E(psi):  
    V_ext=0.5*(rv**2+(gamma*zv)**2) # External potential
    integrand=0.5*np.abs(np.gradient(psi))**2 + V_ext*np.abs(psi)**2 \
        + (gs/2)*np.abs(psi)**4 + (D/2)*Vdd_cyl(psi)*np.abs(psi)**2   
    dr=np.max(r)/len(r)        
    energy =  2*cn.pi*dr*dz*np.sum(integrand)  # Doing the integral
    return energy.real

def Vdd(psi): # Dipole interaction energy
    Rc=25 # Spherical cut-off should be greater than system size
    # Calculating FT of dipole potential
    Uddf=1/3
    Uddf+=1*np.cos(Rc*kmag)/(Rc*kmag)**2
    Uddf=np.nan_to_num(Uddf)
    Uddf-=1*np.sin(Rc*kmag)/(Rc*kmag)**3
    Uddf=np.nan_to_num(Uddf)
    Uddf*=-1*D # FT of the dipole energy. Assumes polarisation along z
    return ifft2(Uddf*fft2(psi**2)) # Convolution theorem

def Vdd_cyl(psi): # Dipole interaction energy with cylindrical cut-off
    Zc=np.max(z)+1 # z cut-off
    theta_k=np.arccos(kz/kmag) # Dipoles polarized in z direction
    # Calculating FT of dipole potential
    Uddf=1/3*(3*(kz/kmag)**2-1)
    Uddf=np.nan_to_num(Uddf)
    Uddf+=np.exp(-Zc*2*np.pi*rho)\
        *(np.sin(theta_k))**2*np.cos(Zc*kz)-np.sin(theta_k)*(kz/kmag)*np.sin(Zc*kz)
    Uddf=np.nan_to_num(Uddf)
    Uddf-= cyl_cutoff.U
    Uddf*=D
    return ifft2(Uddf*fft2(psi**2)) # Convolution theorem


# Pancake potential (Circular potential in r)
#def expVrhpsi):
 #   V=(r>=5).astype(complex)*1e6 # Creates an array which is large for r>5     
  #  V+=gs*np.abs(psi)**2 + Vddr(psi)
   # return np.exp(-0.5j*V*dt)

def expVh(psi): # Harmonic potential
    V=0.5*(rv**2+(gamma*zv)**2) + gs*np.abs(psi)**2/(2*cn.pi) + Vdd(psi)
    return np.exp(-0.5j*V*dt)

# Guess initial psir as gaussian
psir=np.exp(-r**2)

psi=psir.reshape(1,-1).repeat(N,0).T # Creates 2d psi from 1d
#psi=psir.reshape(-1,1).repeat(Nz,1) # Creates 2d psi from 1d
J=J.reshape(-1,1)

### Stability code

dipoles=np.linspace(0,2000,10)
#contacts=np.linspace(0,3000,10)
gammas=np.linspace(1,20,10)

stable_matrix=np.zeros([10,10]) # Stability matrix

for i in range(len(gammas)):
        for j in range(len(dipoles)):
            
            gamma=gammas[i]
            D=dipoles[j]
            
            #dt=dt/gamma # change dt each time for efficiency
            
            # Create a matrix of historical values of psi (to compare each iteration)
            init_psi=np.exp(-r**2)
            init_psi=init_psi/la.norm(init_psi).reshape(1,-1).repeat(N,0).T
            energies=[E(init_psi)]
            
            isConv=False
            p=1
            
            while isConv==False and p<1500: # Loop until convergence or limit reached
                psi=ifft(expTz*fft(expVh(psi)*psi,axis=1),axis=1) # Split step Fourier/Hankel method
                psi=expVh(psi)*hankel.invhankel(expTr*hankel.hankel(psi,J),J)
                psi/=la.norm(psi)
                
                if p%10 == 0: # Will run every 10 iterations
                    
                    # Add energy to list of eneries
                    energies.append(E(psi))
                    
                    m=int(p/10) # Energy index
                    val1=energies[m-1]# First value to consider
                    val2=energies[m] # Second value to consider
                    
                    diff=val2-val1
                    
                    percent_change=100*np.abs((val2-val1))/val1 # Calculate percentage change
                
                    # Checking for convergence     
                    if percent_change < 1: # Will run if the energy changes by less than 1%
                        isConv=True
                        stable_matrix[i][j]=1 # Runs if wavefunction has converged
                    
                
                """ # Convergence based on if wavefunction goes to NaN
                if math.isnan(np.mean(psi)) == True:
                    stable_matrix[i][j]=0 # Runs if wavefunction hasn't converged
                else:
                    stable_matrix[i][j]=1 # Runs if wavefunction has converged
                """    
                p+=1 # Increases iteration number by 1 each loop
                

# Plot results
gammav=gammas.reshape(-1,1).repeat(10,1)
dipolesv=dipoles.reshape(1,-1).repeat(len(dipoles),0)

fig=plt.figure()
ax=plt.axes(projection="3d")
ax.plot_surface(gammav,dipolesv,stable_matrix,cmap="jet")

plt.xlabel("gamma")
plt.ylabel("Dipole strength")
ax.set_title("Convergence plot")
plt.show()
