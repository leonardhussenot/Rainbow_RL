import keras
import numpy as np
import io
import base64
from IPython.display import HTML
import skvideo.io
import cv2
import json

from Environment import *
from Environment_with_hunters import *
from DQN_CNN import *
from Prioritized_DQN import *
from Double_DQN import *
from Dueling_DQN import *
from Multistep_DQN import *
from Distributional_DQN import *
from Human import *
from Noisy_DQN import *

import pickle
# parameters
size = 13
T = 500
temperature = 0.3
epochs_train = 200 # set small when debugging
epochs_test = 10 # set small when debugging


def test(agent,env,epochs,prefix=''):
    # Number of won games
    score = 0

    scores = []
    
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
            action = agent.act(state,train=False)

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
        scores.append(win-lose)

        print("Win/lose count {}/{}. Average score ({})"
              .format(win, lose, score/(1+e)))
    print('Final score: '+str(score/epochs))
    print('Final variance :'+str(np.std(scores)))
    print(scores)
    return score/epochs


def train_validate(agent, env, epoch, prefix='', validation = True):
    # Number of won games
    score = 0
    loss = 0

    validation_list = []

    for e in range(epoch):
        # At each epoch, we restart to a fresh game and get the initial state
        state = env.reset()
        # This assumes that the games will terminate
        game_over = False

        win = 0
        lose = 0

        if (e % 10 == 0) and validation == True :
            validation_list.append(test(agent,env,epochs_test,prefix))


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
                lose = lose - reward

            # Apply the reinforcement strategy
            loss = agent.reinforce(prev_state, state,  action, reward, game_over, e)

        # Save as a mp4
        if e % 10 == 0:
            env.draw(prefix+str(e))

        # Update stats
        score += win - lose

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

    can.add_argument("-a", "--action", default='validate', type=str,
            help="Test or train and validate from scratch")
            
    can.add_argument("-n", "--hunters", default=0, type=int,
            help="Number of hunters")

    the = can.parse_args()




    if the.solver == "dqn":

        env = Environment(grid_size=size, max_time=T, temperature=0.3) if the.hunters == 0 else Environment_with_hunters(grid_size=size, max_time=T, temperature=0.3, hunters=the.hunters)
        agent_cnn = DQN_CNN(size, epsilon = 0.1, memory_size=20000, batch_size = 32, n_state= 3 if the.hunters==0 else 4)

        if the.action == "test":
            agent_cnn.load(name_weights='/dqnmodel.h5',name_model='/dqnmodel.json')
            print('Test of the network')
            test(agent_cnn,env,epochs_test,prefix='dqn')

        if the.action == "validate":
            print('Begin validation')
            train_validate(agent_cnn,env,epochs_train,prefix='dqn')


    if the.solver == "prioritized_dqn":

        env = Environment(grid_size=size, max_time=T, temperature=0.3) if the.hunters == 0 else Environment_with_hunters(grid_size=size, max_time=T, temperature=0.3, hunters=the.hunters)
        agent_cnn = Prioritized_DQN_CNN(size, epsilon = 0.1, memory_size=20000, batch_size = 32, n_state= 3 if the.hunters==0 else 4)

        if the.action == "test":
            agent_cnn.load(name_weights='/prioritized_dqnmodel.h5',name_model='/prioritized_dqnmodel.json')
            print('Test of the network')
            test(agent_cnn,env,epochs_test,prefix='prioritized_dqn')

        if the.action == "validate":
            print('Begin validation')
            train_validate(agent_cnn,env,epochs_train,prefix='prioritized_dqn')


    if the.solver == "double_dqn":

        env = Environment(grid_size=size, max_time=T, temperature=0.3) if the.hunters == 0 else Environment_with_hunters(grid_size=size, max_time=T, temperature=0.3, hunters=the.hunters)
        agent_cnn = Double_DQN_CNN(size, epsilon = 0.1, memory_size=20000, batch_size = 32, n_state= 3 if the.hunters==0 else 4)

        if the.action == "test":
            agent_cnn.load(name_weights='/double_dqnmodel.h5',name_model='/double_dqnmodel.json')
            print('Test of the network')
            test(agent_cnn,env,epochs_test,prefix='double_dqn')

        if the.action == "validate":
            print('Begin validation')
            train_validate(agent_cnn,env,epochs_train,prefix='double_dqn')

    if the.solver == "dueling_dqn":

        env = Environment(grid_size=size, max_time=T, temperature=0.3) if the.hunters == 0 else Environment_with_hunters(grid_size=size, max_time=T, temperature=0.3, hunters=the.hunters)
        agent_cnn = Dueling_DQN_CNN(size, epsilon = 0.1, memory_size=20000, batch_size = 32, n_state= 3 if the.hunters==0 else 4)

        if the.action == "test":
            agent_cnn.load(name_weights='/dueling_dqnmodel.h5',name_model='/dueling_dqnmodel.json')
            print('Test of the network')
            test(agent_cnn,env,epochs_test,prefix='dueling_dqn')

        if the.action == "validate":
            print('Begin validation')
            train_validate(agent_cnn,env,epochs_train,prefix='dueling_dqn')

    if the.solver == "multistep_dqn":

        env = Environment(grid_size=size, max_time=T, temperature=0.3) if the.hunters == 0 else Environment_with_hunters(grid_size=size, max_time=T, temperature=0.3, hunters=the.hunters)
        agent_cnn = Multistep_DQN_CNN(size, epsilon = 0.1, memory_size=20000, batch_size = 32, n_state= 3 if the.hunters==0 else 4)

        if the.action == "test":
            agent_cnn.load(name_weights='/multistep_dqn_n=3model.h5',name_model='/multistep_dqn_n=3model.json')
            print('Test of the network')
            test(agent_cnn,env,epochs_test,prefix='multistep_dqn')

        if the.action == "validate":
            print('Begin validation')
            print('testing different values of n')
            for n in [5]:
                print('----------- {} ----------'.format(n))
                agent_cnn = Multistep_DQN_CNN(size, n=n, epsilon = 0.1, memory_size=20000, batch_size = 32, n_state= 3 if the.hunters==0 else 4)
                train_validate(agent_cnn,env,epochs_train,prefix='multistep_dqn_n={}'.format(n))

    if the.solver == "distributional_dqn":

        env = Environment(grid_size=size, max_time=T, temperature=0.3) if the.hunters == 0 else Environment_with_hunters(grid_size=size, max_time=T, temperature=0.3, hunters=the.hunters)
        agent_cnn = Distributional_DQN_CNN(size, epsilon = 0.1, memory_size=20000, batch_size = 32, n_state= 3 if the.hunters==0 else 4)

        if the.action == "test":
            agent_cnn.load(name_weights='/distributional_dqnmodel.h5',name_model='/distributional_dqnmodel.json')
            print('Test of the network')
            test(agent_cnn,env,epochs_test,prefix='distributional_dqn')

        if the.action == "validate":
            print('Begin validation')
            agent_cnn = Distributional_DQN_CNN(size, epsilon = 0.1, memory_size=20000, batch_size = 32)
            train_validate(agent_cnn,env,epochs_train,prefix='distributional_dqn')

    if the.solver == "human":

        env = Environment(grid_size=size, max_time=T, temperature=0.3) if the.hunters == 0 else Environment_with_hunters(grid_size=size, max_time=T, temperature=0.3, hunters=the.hunters)
        agent_cnn = Human(size, epsilon = 0.1)

        if the.action == "test":
            print('Test of the network')
            test(agent_cnn,env,epochs_test,prefix='human')

        if the.action == "validate":
            print('Begin validation')
            train_validate(agent_cnn,env,epochs_train,prefix='human')
            
            
    if the.solver == "noisy_dqn":

        env = Environment(grid_size=size, max_time=T, temperature=0.3) if the.hunters == 0 else Environment_with_hunters(grid_size=size, max_time=T, temperature=0.3, hunters=the.hunters)
        agent_cnn = Noisy_DQN(size, epsilon = 0.1, n_state= 3 if the.hunters==0 else 4)

        if the.action == "test":
            agent_cnn.load(name_weights='/noisy_dqnmodel.h5',name_model='/noisy_dqnmodel.json')
            print('Test of the network')
            test(agent_cnn,env,epochs_test,prefix='noisy_dqn')

        if the.action == "validate":
            print('Begin validation')
            train_validate(agent_cnn,env,epochs_train,prefix='noisy_dqn')
