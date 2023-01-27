#!/usr/bin/env python
# coding: utf-8

# # Find the treasure game with Q Learning
# 
# Aim of this game was for me to learn Q learning basically. Game is about finding the bigger treasure lying at some point in the one dimensional finite map. 
# 
# Game includes:
# 
# -Two treasures (positive reward), one of them is bigger than the other.
# 
# -If agent drops out of the map it gets punishment (negative reward) and the game finishes.
# 
# -Main goal is for agent to learn its way to the big treasure as fast as possible and maximize the amount of reward it receives.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from random import *


# In[2]:


#Initialize the Q-table
#Every row in the Q table corresponds to a positin on which agent be
#Every columns corresponds to each action
Qtable = np.zeros((21,3)) #For this state, map consists of 21 discrete positions and there are 3 actions that agent take

#Action space, agent can go backwards, forwards or choose the say in the same position 
actions = np.array([-1,0,1])

#Value for epsilon-greedy policy
epsilon = 0.5

#Learning rates
eta = 0.9
gamma = 0.9

#Number of games
num_games = 100

#Number of steps per game
num_steps = 100

#Positions of the treasure
treasure_1 = 10
treasure_2 = 15

#Amount of the treasures
reward_1 = 10
reward_2 = 100

#Initial position
start = 5

for i in range(num_games):
    
    position = start

    x = np.array([position])
    
    total_reward = 0
    
    for j in range(num_steps):

        if random() < epsilon:

            next_action = np.random.choice(actions)

        else:
            
            if Qtable[position][0] == 0 and Qtable[position][1] == 0 and Qtable[position][2] == 0:
                
                next_action = np.random.choice(actions)
            
            else:
                
                next_action = actions[np.argmax(Qtable[position])]

        new_position = position + next_action
        
        reward = 0
         
        if new_position == treasure_1:
            reward = 10
            total_reward += reward_1
            
        elif new_position == treasure_2:
            reward = 100
            total_reward += reward_2
            
        elif new_position == 0:
            
            #Punishment
            reward = -10
            
            #Bellman equations for updating Q-table
            Qtable[position, next_action + 1] = Qtable[position, next_action + 1] + eta*(reward + gamma*(np.max(Qtable[new_position])) - Qtable[position,next_action + 1])
            position = new_position
            x = np.append(x,position)
            break
            
        elif new_position == 20:
            
            reward = -10
            Qtable[position, next_action + 1] = Qtable[position, next_action + 1] + eta*(reward + gamma*(np.max(Qtable[new_position])) - Qtable[position,next_action + 1])
            position = new_position
            x = np.append(x,position)
            break
            
        Qtable[position, next_action + 1] = Qtable[position, next_action + 1] + eta*(reward + gamma*(np.max(Qtable[new_position])) - Qtable[position,next_action + 1])
        position = new_position
        x = np.append(x,position)

    #plt.figure()
    #plt.plot(x)
    
print(Qtable/np.max(Qtable))

