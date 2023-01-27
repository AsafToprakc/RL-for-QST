#!/usr/bin/env python
# coding: utf-8

# In[2]:


from qutip import *
import numpy as np


# In[49]:


#this functions constructs the Hamiltonian for N spin Heisenberg chain
#if N = 1, a chain does not make sense therefore it just returns an empty matrix
def Hamiltonian(N):

    hbar = 1
    
    SpinMatrices = np.zeros((N,3,2**N,2**N), dtype = "complex_")

    Sigmas = 0.5*hbar*np.array([sigmax(),sigmay(),sigmaz()])

    for i in range(N):
        for j in range(3):
            SpinMatrices[i][j] = tensor(qeye(2**(N-1-i)), Qobj(Sigmas[j]),qeye(2**i))


    LadderOp = np.zeros((N,2,2**N,2**N), dtype = "complex_")

    for i in range(N):
        LadderOp[i][0] = SpinMatrices[i,0] - 1j*SpinMatrices[i,1] #down
        LadderOp[i][1] = SpinMatrices[i,0] + 1j*SpinMatrices[i,1] #up

    state = ket(N*"d")

    J = 1

    H = J*np.zeros((2**N,2**N), dtype = "complex_")

    if N == 2:
        H = Qobj(SpinMatrices[0,2])*Qobj(SpinMatrices[1,2]) + 0.5*(Qobj(LadderOp[0,0])*Qobj(LadderOp[1,1]) + Qobj(LadderOp[0,1])* Qobj(LadderOp[1,0]))

    else:
        for i in range(N):

            H += np.array(Qobj(SpinMatrices[i%N,2])*Qobj(SpinMatrices[(i+1)%N,2]))
            H += 0.5*np.array(Qobj(LadderOp[i%N,0])*Qobj(LadderOp[(i+1)%N,1]))
            H += 0.5*np.array(Qobj(LadderOp[i%N,1])*Qobj(LadderOp[(i+1)%N,0]))

    dim = [[],[]]

    for j in range(N):
        dim[0].append(2)
        dim[1].append(2)

    H = Qobj(H, dims = dim)
    
    return H

#this code returs the diagonalized Hamiltonian, eigenvector matrix and eigenvalues respectively
#p.s.: eigenvector matrix and eigenvalue list are ordered i.e. eigenvector in Q[:,i] corresponds to related eigenvalue in EigenValues[i]
def NewHamiltonian(N):
    
    M = Hamiltonian(N)
    eig = M.eigenstates()
    EigenValues  = eig[0]
    EigenVectors = eig[1]
    
    M = M.full()
    
    Q = np.zeros((len(M),len(M)), dtype = 'complex_')
    
    for i in range(len(EigenVectors)):
        
        Q[:,i] = EigenVectors[i].full()[:,0]
       
    Qinv = np.linalg.inv(Q)
    
    NewHamiltonian = np.matmul(np.matmul(Qinv,M),Q)
                
    ChangeOfBasis = Qinv
           
    #NewHamiltonian = np.round(NewHamiltonian,10)
    NewHamiltonian = Qobj(NewHamiltonian).tidyup().full()
    
    dim = [[],[]]
    
    for i in range(N):
        dim[0].append(2)
        dim[1].append(2)
    
    NewHamiltonian = Qobj(NewHamiltonian, dims = dim)
    Q = Qobj(Q, dims = dim)
    
    return NewHamiltonian, Q, EigenValues

#This function return the lowest energy eigenvalue
#If there multiple, then returns all of them, for example N = 3
def MinEnergyEigenvector(N):
    
    M = Hamiltonian(N)
    
    N
    
    eigenvalues = np.round(np.diag(NewHamiltonian(N)[0]),5)
    
    lowest = min(eigenvalues)
    
    indexes = [i for i, x in enumerate(eigenvalues) if x == lowest]
    
    M = Qobj(M)
    R = np.zeros((len(M.full()),len(indexes)), dtype = 'complex_')
    
    for i in range(len(indexes)):
        R[:,i] = np.array(NewHamiltonian(N)[1])[:,indexes[i]]

    dim = [[],[]]
    
    for i in range(N):
        dim[0].append(2)
        dim[1].append(2)
        
    R = Qobj(R, dims = dim)
    
    return R


# In[40]:


Hamiltonian(2)


# In[41]:


NewHamiltonian(2)[0]


# In[42]:


NewHamiltonian(2)[1]


# In[54]:


NewHamiltonian(2)[2]


# In[57]:


MinEnergyEigenvector(2)

