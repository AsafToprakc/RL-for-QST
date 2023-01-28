#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from random import *
from qutip import *
from sympy.physics.quantum import TensorProduct


# In[53]:


def Rotate_X(angle):
    sigma_X = np.array([[0,1],[1,0]])
    return np.matrix(sp.linalg.expm(-1j*0.5*angle*sigma_X))

def Rotate_Y(angle):
    sigma_Y = np.array([[0,-1j],[1j,0]])
    return np.matrix(sp.linalg.expm(-1j*0.5*angle*sigma_Y))

def Rotate_Z(angle):
    sigma_Z = np.array([[1,0],[0,-1]])
    return np.matrix(sp.linalg.expm(-1j*0.5*angle*sigma_Z))

def measure(state,MC):
    
    MC = np.matrix(MC)
    state = np.matrix(state)
    
    MeasurementOperator = MC*MC.H
    
    t = state.H*MeasurementOperator*state
    t = np.real(t[0,0])
    P = np.round(np.array([t, 1 - t]),5)
    
    return np.random.choice([1,-1], p = P)

num_measurement = 10**3

def state_reconstruction(computed_state,index):
    
    MC = np.array([Rotate_Y(np.pi/2)*ket("u"),Rotate_X(-np.pi/2)*ket("u"),ket("u")])
    
    measurement_configuration = MC[index]

    count_up = 0
    count_down = 0
    
    for i in range(num_measurement):

        if measure(computed_state,measurement_configuration) == 1:
            count_up += 1

        else:
            count_down += 1
            
    prob_up = count_up / num_measurement
    prob_down = count_down / num_measurement
    
    c_up = np.sqrt(prob_up)
    c_down = np.sqrt(prob_down)

    reconstructed_state = c_up*measurement_configuration + c_down*Rotate_Y(np.pi)*measurement_configuration
    
    return reconstructed_state


# In[54]:


#State that we want to do the tomography on, for this case it is |+>
computed_state = Rotate_Y(np.pi/2)*ket("u")

#Measurement configurations, i.e. on which axises measurement will be done, X,Y or Z with indexes 0,1,2 respectively
MConfigurations = np.array([0,1,2])
    
#initial Q-table
Qtable = np.zeros(3)

#learning rate
eta = 0.9
gamma = 0.9

#Value for epsilon-greedy policy
epsilon = 0.1

#Number of times different mesaurement configurations will be tried
num_episodes = 10**2

#Number of measurements to construct the state with a chosen measurement configuration
num_measurement = 10

for i in range(num_episodes):
    
    #Choosing the measurement configuration
    
    if random() < epsilon: #exploration
        MConfiguration_index = np.random.choice(np.arange(len(MConfigurations)))
    
    else: #explotation
        
        if all(v == 0 for v in Qtable): #this part exists because Python chooses the first item in the list with np.argmax if list consists of zeros
            MConfiguration_index = np.random.choice(np.arange(len(MConfigurations)))
            
        else:
            MConfiguration_index = np.argmax(Qtable)
    
    #Reconstruction of the computed state with a chosen measurement configuration
    reconstructed_state = state_reconstruction(computed_state, MConfiguration_index)
        
    #Reward for reconstructing the computed state with the chosen measurement configuration, fidelity for now
    reward = np.abs(np.vdot(reconstructed_state,computed_state)[0,0])
    
    #Bellman equation but Q-table consists of only 1 row, hence 1 state, so its definition may be problematic
    Qtable[MConfiguration_index] = Qtable[MConfiguration_index] + eta*(reward + gamma*np.max(Qtable) - Qtable[MConfiguration_index])
    
#Result of the RL algorithm, the best measurement configuration to reconstruct the computed state
MConfigurations[np.argmax(Qtable)]

num_measurement = 10**3
reconstructed_state = state_reconstruction(computed_state,np.argmax(Qtable))

#Comparison
print(computed_state)
print(reconstructed_state)

#Comparison without the phase term
print(np.abs(computed_state))
print(np.array(np.abs(reconstructed_state)))

