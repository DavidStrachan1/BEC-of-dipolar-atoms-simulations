# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:19:30 2020

Modelling a dipolar Bose-Einstein condensate by finding the ground state solution 
of the Gross–Pitaevskii equation using imaginary time propagation
and the split step Hankel method in 3D with cylindrical symmetry
"""
import numpy as np
import math
from numpy.fft import fft,ifft,fftfreq
from numpy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from tqdm import tqdm
import scipy.constants as cn
from hankel import hankel_class
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

def V_box(r,z,Rc,gamma_z):
    return (r/(Rc-0.2))**60 + 0.5*(gamma_z*z)**2

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
    ax=plt.figure()
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
    ax.plot_surface(r[0:n_1,z_coord_i:z_coord_f],z[0:n_1,z_coord_i:z_coord_f],np.abs(psi)[0:n_1,z_coord_i:z_coord_f],cmap='jet')
    
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
#gamma=7 # wz/w trap aspect ratio

#Dimensionless unit formulas
xs=np.sqrt(cn.hbar/(m*w)) # Length scaling parameter
gs=4*cn.pi*a*Na/xs # Dimensionless contact coefficient
D=m*Na*Cdd/(4*cn.pi*(cn.hbar)**2*xs) # Dimensionless dipolar interaction parameter

#Override of parameters:
#D=100
gs=0

#Numerical simulation parameters
N=300
max_r=10
max_z_init=10
dt=-0.01j # Imaginary time propagation unit
Nit=1000 # Number of iterations
Rc=10 # radius of sphere cutoff (Not needed in r direction)
Zc=6 # z cutoff

# Define grids
hankel=hankel_class(N,max_r) # Create instance of hankel transform with set parameters

### Stability stuff

size=10

dipoles=np.linspace(0,180,size)
#contacts=np.linspace(0,3000,10)
gammas=np.linspace(1,20,size)

#dipoles=[0,30]
#gammas=[1]

dipoles=[70]
gammas=[7.3]

stable_matrix=np.zeros([size,size])+2 # Stability matrix (starts off assuming no convergence)

for i in range(len(gammas)):
    
    gamma=gammas[i]
    
    max_z=max_z_init*gamma**(-0.5) # Changes max_z as a function of gamma (for higher accuracy)
    Zc=max_z/2
        
    r,rho = create_grid(N,max_r,max_z,hankel,True)
    z,kz = create_grid(N,max_r,max_z,hankel,False)
                
    k_rho=2*np.pi*rho # rho is wavevector in hankel space
    J=hankel.J.reshape(-1,1) # Used in hankel transform
                
    # Magnitude of k^2
    kmag=(k_rho**2+kz**2)**0.5
        
    # Kinetic energy operator and the exponential form
    T=0.5*kmag**2
    expT=np.exp(-1j*T*dt)
        
    # Harmonic potential
    V=V_harmonic(r,z,1,gamma)
    #V=V_box(r,z,max_r,gamma)
    
    dipoleConv=True
    j=0
    while dipoleConv == True and j<len(dipoles):
                
        D=dipoles[j]
        
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
        
        # Guess initial psi as gaussian
        #psi=np.exp(-(r**2/5+gamma*z**2))
        psi=np.exp(-(r**2+gamma*z**2)/50)
        psi/=norm(psi,r,z,N,max_z)
            
        # Initial list of energies
        energies=[E(psi,N,max_r,max_z,V,Uddf,gs,Na,kmag,hankel,J)]

        hasEnded=False
        p=1
        
        ## dt based on energy
        
        # take energy values every 1 for large N
        
        # Imaginary time propagation
        
        while hasEnded==False and p<Nit: # Loop until convergence or limit reached
                psi=expVh(psi,V,gs,Uddf,dt,J)*psi # Split step Fourier/Hankel method
                psi=expT*fft(hankel.hankel(psi,J),axis=1)
                psi/=norm(psi,r,z,N,max_z)
                psi=ifft(hankel.invhankel(psi,J),axis=1)
                psi/=norm(psi,r,z,N,max_z)
                psi*=expVh(psi,V,gs,Uddf,dt,J)
                psi/=norm(psi,r,z,N,max_z)
                
                
                if p%10 == 0: # Will run every 10 iterations
                    # Add energy of psi to an existing list
                    energies.append(E(psi,N,max_r,max_z,V,Uddf,gs,Na,kmag,hankel,J))                  
                        
                    ## Convergence based on if wavefunction goes to NaN first
                    if math.isnan(np.mean(psi)) == False:
                        
                        val1=psi[0,int(N/2)] # Value of psi at r=0 and middle of z
                        val2=psi[:,int(N/2)].max() # Max value of psi along middle of z
                        max_psi_percent_diff=100*(val2-val1)/val1
                        max_index=np.where(psi==psi.max())[0][0] # Index of max value
                        # Checking for red blood cell by if centre value of psi is less than the max
                        if max_psi_percent_diff > 10 and max_psi_percent_diff<100 and max_index<N/2: # Peak needs to be near centre for red blood cell
                            stable_matrix[i][j]=1 # Runs if red blood cell
                        
                        # Check energy
                        m=int(p/10) # Energy index
                        
                        if m>5: # Will run once 5 energies have been taken
                            
                            energy_array=np.array(energies[-5:]) # Takes last 5 energy values
                        
                            energy_deviation=energy_array.ptp()/energy_array.mean() # Range over mean
                            
                            # Checking for energy convergence     
                            if energy_deviation < 1e-10: # Will run if the energy changes by a small amount
                                stable_matrix[i][j]=0
                                #print("energy stable")
                                hasEnded=True
                                
                                ## Fix issue for convergence for gamma=3,D=70
                                
                            # Checks for a big spike in energy    
                            if np.abs(np.gradient(energy_array)).max() > 100:
                                stable_matrix[i][j]=2
                                hasEnded=True
                            
                    elif math.isnan(np.mean(psi)) == True:
                        stable_matrix[i][j]=2 # Runs if wavefunction hasn't converged
                        hasEnded=True             
                
                p+=1 # Increases iteration number by 1 each loop
                
        if j >= 3:
            if (stable_matrix[i][j-2:j+1] == [2,2,2]).all(): # Runs if no convergence 3 times in a row
                dipoleConv=False
                stable_matrix[i][j:] = [2]*len(stable_matrix[i][j:]) # Sets rest of dipoles to no convergence
                
        print("%g, %g, %d" %(round(gamma,1),round(D,1),stable_matrix[i][j]))
        
        j+=1 # Moves to next dipole
                
gammas=np.array(gammas)
dipoles=np.array(dipoles)

# Plot heatmap
dipole_columns=np.max(dipoles)-dipoles
ax = sns.heatmap(np.rot90(stable_matrix), xticklabels=gammas, yticklabels=dipole_columns);
ax.set_xticks(ax.get_xticks()[::2].round(1));
ax.set_xticklabels(gammas[::2].round(1),rotation=45, horizontalalignment='right');
ax.set_yticks(ax.get_yticks()[::2].astype(int));
ax.set_yticklabels(dipole_columns[::2].astype(int),rotation=45, horizontalalignment='right');
plt.xlabel("$\gamma$")
plt.ylabel("D")
plt.show()

plt.plot(r[:,0],psi[:,int(N/2)]);
