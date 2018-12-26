import keras
import numpy as np
import io
import base64
from IPython.display import HTML
import skvideo.io
import cv2
import json

from Environment import *
from DQN_CNN import *
from Prioritized_DQN import *
from Double_DQN import *

import pickle 
# parameters
size = 13
T=500
temperature=0.3
epochs_train=101 # set small when debugging
epochs_test=10 # set small when debugging

			 
def test(agent,env,epochs,prefix=''):
    # Number of won games
    score = 0
        
    for e in range(epochs):
    ##### FILL IN HERE
    # At each epoch, we restart to a fresh game and get the initial state
        state = env.reset()
    # This assumes that the games will end
        game_over = False

        win = 0
        lose = 0

        while not game_over:
        # The agent performs an action
            action = agent.act(state)

        # Apply an action to the environment, get the next state, the reward
        # and if the games end
            prev_state = state
            state, reward, game_over = env.act(action)

        # Update the counters
            if reward > 0:
                win = win + reward
            if reward < 0:
                lose = lose -reward

        # Apply the reinforcement strategy
            #loss = agent.reinforce(prev_state, state,  action, reward, game_over, e)
          
        # Save as a mp4
        env.draw(prefix+str(e))

        # Update stats
        score = score + win-lose

        print("Win/lose count {}/{}. Average score ({})"
              .format(win, lose, score/(1+e)))
    print('Final score: '+str(score/epochs))
    return score/epochs
	
	
def train_validate(agent,env,epoch,prefix='', validation = True):
    # Number of won games
    score = 0
    loss = 0

    for e in range(epoch):
        # At each epoch, we restart to a fresh game and get the initial state
        state = env.reset()
        # This assumes that the games will terminate
        game_over = False

        win = 0
        lose = 0
        
        validation_list = []
        
        if (e % 10 == 0) and validation == True :
            validation_list.append(test(agent,env,10,prefix))
            

        while not game_over:
            # The agent performs an action
            action = agent.act(state)

            # Apply an action to the environment, get the next state, the reward
            # and if the games end
            prev_state = state
            state, reward, game_over = env.act(action)

            # Update the counters
            if reward > 0:
                win = win + reward
            if reward < 0:
                lose = lose -reward

            # Apply the reinforcement strategy
            loss = agent.reinforce(prev_state, state,  action, reward, game_over, e)

        # Save as a mp4
        if e % 10 == 0:
            env.draw(prefix+str(e))

        # Update stats
        score += win-lose

        print("Epoch {:03d}/{:03d} | Loss {:.4f} | Win/lose count {}/{} ({})"
              .format(e, epoch, loss, win, lose, win-lose))
        agent.save(name_weights=prefix+'model.h5',name_model=prefix+'model.json')
        
    with open('./models/'+prefix, 'wb') as f:
        pickle.dump(validation_list, f)    
	
    
    
            
		
if __name__=="__main__":
    import argparse
    can = argparse.ArgumentParser()

    can.add_argument("-s", "--solver", default='dqn', type=str,
            help="name of the solver")
            
    can.add_argument("-a", "--action", default='validation', type=str,
            help="Test or train and validate from scratch")

    the = can.parse_args()
     
       
          
	
    if the.solver == "dqn":
	
	    env = Environment(grid_size=size, max_time=T,temperature=0.3)
	    agent_cnn = DQN_CNN(size, epsilon = 0.1, memory_size=20000, batch_size = 32)
		
	    if the.action == "test":
	        agent_cnn.load(name_weights='./models/dqnmodel.h5',name_model='./models/dqnmodel.json')
	        print('Test of the network')
	        test(agent_cnn,env,epochs_test,prefix='dqn')
		
	    if the.action == "validate":
	        print('Begin validation')
	        train_validate(agent_cnn,env,epochs_train,prefix='dqn')
            
            
    if the.solver == "prioritized_dqn":
	
	    env = Environment(grid_size=size, max_time=T,temperature=0.3)
	    agent_cnn = Prioritized_DQN_CNN(size, epsilon = 0.1, memory_size=20000, batch_size = 32)
		
	    if the.action == "test":
	        agent_cnn.load(name_weights='./models/prioritized_dqnmodel.h5',name_model='./models/prioritized_dqnmodel.json')
	        print('Test of the network')
	        test(agent_cnn,env,epochs_test,prefix='prioritized_dqn')
		
	    if the.action == "validate":
	        print('Begin validation')
	        train_validate(agent_cnn,env,epochs_train,prefix='prioritized_dqn')
            
            
    if the.solver == "double_dqn":
	
	    env = Environment(grid_size=size, max_time=T,temperature=0.3)
	    agent_cnn = Double_DQN_CNN(size, epsilon = 0.1, memory_size=20000, batch_size = 32)
		
	    if the.action == "test":
	        agent_cnn.load(name_weights='./models/double_dqnmodel.h5',name_model='./models/double_dqnmodel.json')
	        print('Test of the network')
	        test(agent_cnn,env,epochs_test,prefix='double_dqn')
		
	    if the.action == "validate":
	        print('Begin validation')
	        train_validate(agent_cnn,env,epochs_train,prefix='double_dqn')
			
		
	
