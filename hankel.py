# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:19:01 2020

@author: ganda

Discrete hankel transform using dini expansion (see paper for method)
"""
import numpy as np
import scipy.special as sp

class hankel_class:
    
    Nj=300 # Size of hankel space
    b=20 # Max radius of r
    
    Jprimezeros=sp.jnp_zeros(0,Nj) # Array of zeros of J0 prime
    Jprimezeros=np.insert(Jprimezeros,0,0) # First root is 0
    
    # Calculate S
    k=int(Nj/4)
    ak=Jprimezeros[k]
    S_sum=0
    for n in range(1,Nj+1):
        an=Jprimezeros[n]
        S_sum+=sp.j0(an)**-2*sp.j0((ak*an)/sp.jn_zeros(0,Nj+1)[Nj])**2
    S=2*np.abs(sp.j0(ak)**-1)*np.sqrt(1+S_sum)
    
    r=Jprimezeros*b/S
    beta=S/(2*np.pi*b)
    rho=Jprimezeros*beta/S
    
    
    # Calculating the orthogonal matrix C
    C=np.zeros((Nj+1,Nj+1))
    for n in range(Nj+1):
        an=Jprimezeros[n]
        for m in range(Nj+1):
            am=Jprimezeros[m]
            C[n][m]=2/S*np.abs(sp.j0(an)**-1)*np.abs(sp.j0(am)**-1)*sp.j0(an*am/S)
        
       
    def G(g,J):
        return g*hankel_class.beta/J
    
    def F(f,J):
        return f*hankel_class.b/J
    
    def J():
        return np.abs(sp.j0(hankel_class.Jprimezeros))

    def r():
        return hankel_class.Jprimezeros*hankel_class.b/hankel_class.S
    
    def rho():
        return hankel_class.Jprimezeros*hankel_class.beta/hankel_class.S
    
    def hankel(func,J):
        out=np.matmul(hankel_class.C,hankel_class.F(func,J))
        return out*J/hankel_class.beta
    
    def invhankel(func,J):
        out=np.matmul(hankel_class.C,hankel_class.G(func,J))
        return out*J/hankel_class.b
