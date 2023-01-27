#!/usr/bin/env python
# coding: utf-8

# In[188]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from random import *
from qutip import *


# In[189]:


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
    MeasurementVector = np.matrix(MC)
    state = np.matrix(state)
    
    MeasurementOperator = MeasurementVector*MeasurementVector.H
    
    t = state.H*MeasurementOperator*state
    t = np.real(t[0,0])
    P = np.round(np.array([t, 1 - t]),5)
    
    return np.random.choice([1,-1], p = P)

def state_reconstruction(computed_state,theta):
    
    measurement_configuration = Rotate_Y(theta)*ket("u").full()

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


# In[207]:


#State that we want to do the tomography on, for this case it is |+>
computed_state = Rotate_Y(np.pi/2)*ket("u")

#Number of possible measurement configurations
#p.s.: if you want to construct the state exactly make sure that the axis (or angle or measurement configuration) of the computed state is in the list of possible measurement configurations
L = 16

#Measurement configurations, i.e. different axises on the XZ plane of Bloch sphere on which we can perform a number of measurements
MConfigurations = np.linspace(0,1,L + 1)[:-1]*np.pi

#initial Q-table
Qtable = np.zeros(L, dtype = "complex_")

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
    reconstructed_state = state_reconstruction(computed_state, MConfigurations[MConfiguration_index])
        
    #Reward for reconstructing the computed state with the chosen measurement configuration, fidelity for now
    reward = np.abs(np.vdot(reconstructed_state,computed_state)[0,0])
    
    #Bellman equation but Q-table consists of only 1 row, hence 1 state, so its definition may be problematic
    Qtable[MConfiguration_index] = Qtable[MConfiguration_index] + eta*(reward + gamma*np.max(Qtable) - Qtable[MConfiguration_index])
    
#Result of the RL algorithm, the best measurement configuration to reconstruct the computed state
MConfigurations[np.argmax(Qtable)]

num_measurement = 10**3

#Comparison
print(computed_state)
print(state_reconstruction(computed_state,MConfigurations[np.argmax(Qtable)]))

