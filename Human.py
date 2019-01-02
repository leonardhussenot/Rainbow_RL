import numpy as np
import io
import base64
from IPython.display import HTML
import skvideo.io
import cv2
import json
import os

from Agent import *
from Memory import *

class Human(Agent):
    def __init__(self, grid_size, epsilon):
        super(Human, self).__init__(epsilon = epsilon)
        self.grid_size = grid_size

    def learned_act(self, s):
        return self.choose_action(s)

    def reinforce(self, s_, n_s_, a_, r_, game_over_, epoch_):
        return 0
        
    def choose_action(self,s):
        states = [s[3,2,0],s[1,2,0],s[2,3,0],s[2,1,0]]
        legit = [s[3,2,1],s[1,2,1],s[2,3,1],s[2,1,1]]
        visited = [s[3,2,2],s[1,2,2],s[2,3,2],s[2,1,2]]
        
        no_poison=[]
        for i in range(len(states)):
            if states[i]>0 and legit[i]!=-1:
                return i
            else :
                if states[i]==0 and legit[i]!=-1:
                    no_poison.append(i)
                    
        if len(no_poison)==0:
            action = 0  
            while True :
                if legit[action] != -1:
                    r = np.random.randint(4)
                    if r==0:
                        return action
                action += 1
                action = action % len(no_poison)
        else :
            indice = 0
            for i in range(4):
                if visited[no_poison[indice]] ==0:
                    return no_poison[indice]
                    indice += 1
                    indice = indice % len(no_poison)                    
            while True :
                r = np.random.randint(8)
                if r==0:
                    return no_poison[indice]
                indice += 1
                indice = indice % len(no_poison)
                    
    def save(self,name_weights='model.h5',name_model='model.json'):
        print('humans dont save weights')
        

    def load(self,name_weights='model.h5',name_model='model.json'):
        print('humans dont save weights')
        
        
        