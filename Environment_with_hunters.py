import keras
import numpy as np
import io
import base64
from IPython.display import HTML
import skvideo.io
import cv2
import json
import os


## Abstract
class Environment_with_hunters(object):
    def __init__(self):
        pass

    def act(self, act):
        """
        One can act on the environment and obtain its reaction:
        - the new state
        - the reward of the new state
        - should we continue the game?

        :return: state, reward, game_over
        """
        pass


    def reset(self):
        """
        Reinitialize the environment to a random state and returns
        the original state

        :return: state
        """
        pass

    def draw(self):
        """
        Visualize in the console or graphically the current state
        """
        pass

## Cheese and rat

class Environment_with_hunters(object):
    def __init__(self, grid_size=10, max_time=500, temperature=0.1, hunters=0):
        grid_size = grid_size+4
        self.grid_size = grid_size
        self.max_time = max_time
        self.temperature = temperature

        #board on which one plays
        self.board = np.zeros((grid_size,grid_size))
        self.position = np.zeros((grid_size,grid_size))
        self.trajectory = np.zeros((grid_size,grid_size))
        self.board_with_hunters = np.zeros((grid_size,grid_size))

        # coordinate of the cat
        self.x = 0
        self.y = 1
        
        # coordinates od the hunter
        self.hunters = hunters
        self.h_x = []
        self.h_y = []
        for i in range(self.hunters):
            x = np.random.randint(3, self.grid_size-3, size=1)[0]
            y = np.random.randint(3, self.grid_size-3, size=1)[0]
            self.h_x.append(x)
            self.h_y.append(y)
         

        # self time
        self.t = 0

        self.scale=16

        self.to_draw = np.zeros((max_time+2, grid_size*self.scale, grid_size*self.scale, 3))


    def draw(self,e):
        skvideo.io.vwrite('videos/'+str(e) + '.mp4', self.to_draw)

    def get_frame(self,t):
        b = np.zeros((self.grid_size,self.grid_size,3))+128
        b[self.board>0,0] = 256
        b[self.board < 0, 2] = 256
        b[self.x,self.y,:]=256
        
        for i in range(len(self.h_x)):
            b[self.h_x[i],self.h_y[i],1]=256
            b[self.h_x[i],self.h_y[i],0]=0
            b[self.h_x[i],self.h_y[i],2]=0
        
        b[-2:,:,:]=0
        b[:,-2:,:]=0
        b[:2,:,:]=0
        b[:,:2,:]=0

        b =  cv2.resize(b, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)

        self.to_draw[t,:,:,:]=b
        
    def hunter_move(self):
        for i in range(len(self.h_x)):
            will_move=np.random.randint(0, 2, size=1)[0]==0
            if will_move:
                action =np.random.randint(0, 4, size=1)[0]
                if action == 0:
                    if self.h_x[i] == self.grid_size-3:
                        self.h_x[i] = self.h_x[i]-1
                    else:
                        self.h_x[i] = self.h_x[i] + 1
                elif action == 1:
                    if self.h_x[i] == 2:
                        self.h_x[i] = self.h_x[i]+1
                    else:
                        self.h_x[i] = self.h_x[i]-1
                elif action == 2:
                    if self.h_y[i] == self.grid_size - 3:
                        self.h_y[i] = self.h_y[i] - 1
                    else:
                        self.h_y[i] = self.h_y[i] + 1
                elif action == 3:
                    if self.h_y[i] == 2:
                        self.h_y[i] = self.h_y[i] + 1
                    else:
                        self.h_y[i] = self.h_y[i] - 1


    def act(self, action):
        """This function returns the new state, reward and decides if the
        game ends."""

        self.get_frame(int(self.t))

        self.position = np.zeros((self.grid_size, self.grid_size))

        self.position[0:2,:]= -1
        self.position[:,0:2] = -1
        self.position[-2:, :] = -1
        self.position[:, -2:] = -1

        self.position[self.x, self.y] = 1
        if action == 0:
            if self.x == self.grid_size-3:
                self.x = self.x-1
            else:
                self.x = self.x + 1
        elif action == 1:
            if self.x == 2:
                self.x = self.x+1
            else:
                self.x = self.x-1
        elif action == 2:
            if self.y == self.grid_size - 3:
                self.y = self.y - 1
            else:
                self.y = self.y + 1
        elif action == 3:
            if self.y == 2:
                self.y = self.y + 1
            else:
                self.y = self.y - 1
        else:
            RuntimeError('Error: action not recognized')

        self.t = self.t + 1
        reward = self.board[self.x, self.y]
        
        self.hunter_move()
        removals = []
        for i in range(len(self.h_x)):
            if self.x == self.h_x[i] and self.y == self.h_y[i]:
                reward -= 100
                removals.append(i)
                
        for i in sorted(removals, reverse=True):
            del self.h_x[i]
            del self.h_y[i]
                
        self.board[self.x, self.y] = 0
        self.board_with_hunters[:,:] = 0
        
        for i in range(len(self.h_x)):
            self.board_with_hunters[self.h_x[i],self.h_y[i]] = -100
            
        self.trajectory[self.x,self.y] = 1
        game_over = self.t > self.max_time
        state = np.concatenate((self.board.reshape(self.grid_size, self.grid_size,1),
                        self.position.reshape(self.grid_size, self.grid_size,1),
                        self.trajectory.reshape(self.grid_size, self.grid_size,1),
                        self.board_with_hunters.reshape(self.grid_size, self.grid_size,1)),axis=2)
        state = state[self.x-2:self.x+3,self.y-2:self.y+3,:]

        return state, reward, game_over

    def reset(self):
        """This function resets the game and returns the initial state"""

        self.x = np.random.randint(3, self.grid_size-3, size=1)[0]
        self.y = np.random.randint(3, self.grid_size-3, size=1)[0]
        
        self.h_x = []
        self.h_y = []
        for i in range(self.hunters):
            x = np.random.randint(3, self.grid_size-3, size=1)[0]
            y = np.random.randint(3, self.grid_size-3, size=1)[0]
            self.h_x.append(x)
            self.h_y.append(y)

        self.trajectory = np.zeros((self.grid_size,self.grid_size))

        bonus = 0.5 * np.random.binomial(1,self.temperature,size=self.grid_size**2)
        bonus = bonus.reshape(self.grid_size,self.grid_size)

        malus = -1.0 * np.random.binomial(1,self.temperature,size=self.grid_size**2)
        malus = malus.reshape(self.grid_size, self.grid_size)

        self.to_draw = np.zeros((self.max_time+2, self.grid_size*self.scale, self.grid_size*self.scale, 3))


        malus[bonus>0]=0

        self.board = bonus + malus

        self.position = np.zeros((self.grid_size, self.grid_size))
        self.position[0:2,:]= -1
        self.position[:,0:2] = -1
        self.position[-2:, :] = -1
        self.position[:, -2:] = -1
        self.board[self.x,self.y] = 0
        self.t = 0
        
        self.board_with_hunters[:,:] = 0
        
        for i in range(self.hunters):
            self.board_with_hunters[self.h_x[i],self.h_y[i]] = -100
            

        global_state = np.concatenate((
                               self.board.reshape(self.grid_size, self.grid_size,1),
                        self.position.reshape(self.grid_size, self.grid_size,1),
                        self.trajectory.reshape(self.grid_size, self.grid_size,1),
                        self.board_with_hunters.reshape(self.grid_size, self.grid_size,1)),axis=2)

        state = global_state[self.x - 2:self.x + 3, self.y - 2:self.y + 3, :]
        return state


# display videos
def display_videos(name):
    video = io.open('videos/'+name, 'r+b').read()
    encoded = base64.b64encode(video)
    return '''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))
