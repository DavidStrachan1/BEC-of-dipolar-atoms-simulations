# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:19:01 2020

@author: ganda

Discrete hankel transform using dini expansion (see paper for method)
"""
import numpy as np
import scipy.special as sp

class hankel_class:
    
    def __init__(self,Nj=300,b=20): # Runs when hankel_class is called
        self.Nj=Nj # Default Nj=300
        self.b=b # Default b=20
    
        Jprimezeros=sp.jnp_zeros(0,self.Nj) # Array of zeros of J0 prime
        Jprimezeros=np.insert(Jprimezeros,0,0) # First root is 0
        self.Jprimezeros=Jprimezeros
        
        J=np.abs(sp.j0(Jprimezeros))
        self.J=J
        
        # Calculate S
        k=int(self.Nj/4)
        ak=Jprimezeros[k]
        S_sum=0
        for n in range(1,self.Nj+1):
            an=Jprimezeros[n]
            S_sum+=sp.j0(an)**-2*sp.j0((ak*an)/sp.jn_zeros(0,self.Nj+1)[self.Nj])**2
        S=2*np.abs(sp.j0(ak)**-1)*np.sqrt(1+S_sum)
        self.S=S
        
        #global r, rho
        r=Jprimezeros*self.b/S
        beta=S/(2*np.pi*self.b)
        rho=Jprimezeros*beta/S
        self.beta=beta
        
        
        # Calculating the orthogonal matrix C
        C=np.zeros((self.Nj+1,self.Nj+1))
        for n in range(self.Nj+1):
            an=Jprimezeros[n]
            for m in range(self.Nj+1):
                am=Jprimezeros[m]
                C[n][m]=2/S*np.abs(sp.j0(an)**-1)*np.abs(sp.j0(am)**-1)*sp.j0(an*am/S)
        self.C=C
        
    
    def J(self):
        return np.abs(sp.j0(self.Jprimezeros))

    def r(self):
        return self.Jprimezeros*self.b/self.S
    
    def rho(self):
        return self.Jprimezeros*self.beta/self.S
    
    def hankel(self,func,_J): # Named _J to not confuse with J in the class
        def F(f):
            return f*self.b/_J
            
        out=np.matmul(self.C,F(func))
        return out*_J/self.beta
    
    def invhankel(self,func,_J):
        def G(g):
            return g*self.beta/_J
        
        out=np.matmul(self.C,G(func))
        return out*_J/self.b
