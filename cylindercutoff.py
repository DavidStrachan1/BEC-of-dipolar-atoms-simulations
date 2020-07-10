# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:44:43 2020

@author: ganda

Cylindrical cut-off for dipole interaction (calculates integral, called U)
"""
import numpy as np
import scipy.special as sp
import scipy.constants as cn
from numpy.fft import fftfreq
from hankel import hankel_class as hankel
from tqdm import tqdm

class cyl_cutoff:
    
    U=0
     
    def main():
        
        # Defining variables being used
        N=301
        r=hankel.r()
        z=np.linspace(-10,10,N)
        dz=(z[N-1]-z[0])/(N)
        z=z.reshape(1,-1).repeat(N,0)
        rho=hankel.rho().reshape(-1,1).repeat(N,1)
       
        kz=fftfreq(N,dz/(2*cn.pi)).reshape(1,-1).repeat(N,0)
            
        rhoc=np.max(r)/2 # r cutoff
        Zc=np.max(z)+1
        nx=ny=100
        dx=0.1
        dy=Zc/ny
    
        # x and y are the rho and z space being integrated over
        x=np.linspace(rhoc,rhoc+nx*dx,nx).reshape(1,-1).repeat(nx,0)
        y=np.linspace(0,Zc,ny).reshape(-1,1).repeat(ny,1)
        
        # f is the function being integrated
        def f(rho,kz):
            return x*np.cos(kz*y)*(x**2-2*y**2)*(x**2+y**2)**(-5/2)*sp.j0(2*cn.pi*rho*x)
        
        # Calculating the integral
        cyl_cutoff.U=np.zeros((N,N))  
        for i in tqdm(range(N),leave=False):
            for j in range(N):
                cyl_cutoff.U[i,j]=dx*dy*np.sum(f(rho[i,0],kz[0,j]))  
    

if __name__ == "__main__":
    cyl_cutoff.main() # Runs code to create U