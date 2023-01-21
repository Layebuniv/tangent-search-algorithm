# % Tangent search algorithm version of the paper "Tangent search algorithm for solving optimization problems"
# %Author, inventor and programmer: Layeb Abdesslem
# %  e-Mail: abdesslem.layeb@univ-constantine2.dz
# % Layeb, Abdesslem. "Tangent search algorithm for solving optimization problems"
# % Neural Computing and Applications (2022): 1-32.

import random
import math
from typing import List

def rastrigin(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_value


import numpy as np
np.random.seed(seed=None)

#---------problem parameters--------------------
dim=30
lb=-5.12
ub=5.12
funcname='rastrigin'

fun = eval(funcname)
lu = np.array([lb * np.ones(dim), ub * np.ones(dim)])

#------------sTSA parameters
pop_size=50
Pswitch=0.3; # the intensification/exploration probability
Pesc=0.0;    # the escape procedure probability

MAX_FES=50000
FES=0

#population creation
from numpy import matlib as mb
pop = mb.repmat(lu[0, :], pop_size,1) + np.random.rand(pop_size, dim) * (mb.repmat(lu[1, :] - lu[0, :], pop_size,1))
# population evaluttion
fitness_pop=np.zeros(pop_size)
for i in range(pop_size):
   X=pop[i,:]
   fitness=fun(X)
   fitness_pop[i]=fitness
   FES=FES+1
   if i==0:
        Best_agent=pop[i,:]
        Best_fit=fitness_pop[i]

   if fitness_pop[i]< Best_fit:
        Best_agent=X
        Best_fit=fitness_pop[i]
        print("best value=",Best_fit, " FES=", FES)
       
    
    #------------------ begin iteration process
    
while FES<=MAX_FES:
    
    for j in range(pop_size):
        X=pop[j,:]
        
        if np.random.rand() <=Pswitch:
           #-----------------exploration phase----------------
            if np.random.rand() <=0.25: # high exploration
                X=X+np.tan(np.random.rand(dim)*np.pi) # large tangent flight
            else:   #low exploration
                if (Best_agent==X).all():
                    teta=np.random.rand(dim)*np.pi/2.5
                    step=0.5*np.sign(np.random.rand(dim)-0.5)#/np.log(1+FES)
                    X=X+step*np.tan(teta)
                else:
                    teta=np.random.rand(dim)*np.pi/2.5 # small tangent flight
                    step=0.5*np.sign(np.random.rand(dim)-0.5)*np.linalg.norm(Best_agent-X)
                    X=X+step*np.tan(teta)
         
        else:   
            #-----------------  intensification phase----------------
            X = pop[j, :] 
            teta = np.random.rand()*np.pi/2.5
            step = np.sign(np.random.rand()-0.5)*np.linalg.norm(Best_agent)*np.log(1+15*dim/FES)
            if np.array_equal(Best_agent, X):
                X = Best_agent + step*(np.tan(teta))*(np.random.rand()*(Best_agent-X))
            else:
                X = Best_agent + step*(np.tan(teta))*(Best_agent-X)
               
        Xnew = pop[j, :] 
        id=np.random.randint(dim)
        ind=np.asarray((np.where(np.random.rand(dim)<=0.2))).flatten()
        ind=np.append(ind,id) 
        Xnew[ind]=X[ind]                                        
        Xnew = X

        Xnew[Xnew>ub]=np.random.rand()*(ub - lb) + lb
        Xnew[Xnew<lb]=np.random.rand()*(ub - lb) + lb
        fitness = fun(Xnew)

        if fitness < fitness_pop[j]:
            pop[j, :] = Xnew
            fitness_pop[j] = fitness
            if fitness < Best_fit:
                Best_agent = Xnew
                Best_fit = fitness
                print("best value=",Best_fit, "  FES=", FES)
        FES += 1
        if FES > MAX_FES:
            break
             

#----------------- escape local procedure (optional )
#     if np.random.rand() <Pesc:
#         im=np.random.randi(pop_size) #  select randomly one search agent
#         X= pop[im, :]    
#         if rand<0.5 :
#             f1 = -1+(1-(-1))*np.random.rand()
#             step = 15*f1/np.log(1+FES)
#             X = X + step*(Best_agent-np.random.rand()*(Best_agent-X))
#         else:
#             teta=np.random.rand()*np.pi;
#             X =  X + np.tan(teta)*(ub-lb)  # generate a random solution by tangent flight
#         Xnew = pop[im, :] 
#         id=np.random.randint(dim)
#         ind=np.asarray((np.where(np.random.rand(dim)<=0.2))).flatten()
#         ind=np.append(ind,id) 
#         Xnew[ind]=X[ind]                                        
#         Xnew = X
#             #-------------evaluation 
#         Xnew[Xnew>ub]=np.random.rand()*(ub - lb) + lb
#         Xnew[Xnew<lb]=np.random.rand()*(ub - lb) + lb
#         fitness = fun(Xnew)

#         if fitness < fitness_pop[j]:
#             pop[j, :] = Xnew
#             fitness_pop[j] = fitness
#             if fitness < Best_fit:
#                 Best_agent = Xnew
#                 Best_fit = fitness
#                 print("best value=",Best_fit, "  iteration=", FES)
#             FES += 1
#             if FES > MAX_FES:
#                 break        

print(Best_fit, funcname)