import keras
import numpy as np
import io
import base64
from IPython.display import HTML
import skvideo.io
import cv2
import json
import os
import math

from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.optimizers import sgd, Adam
from keras.layers import Conv2D, MaxPooling2D, Activation, AveragePooling2D,Reshape,BatchNormalization, Input

from Agent import *
from Memory import *

class Distributional_DQN(Agent):
    def __init__(self, grid_size, discount = 0.9, epsilon = 0.1, memory_size=100, batch_size = 16, n_state=3):
        super(Distributional_DQN, self).__init__(epsilon = epsilon)

        # Initialize Atoms
        self.num_atoms = 51 # From paper C51
        self.v_max = 25     # Maximum reward possible
        self.v_min = -15    # Minimum reward possible
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.support = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        # Discount for Q learning
        self.discount = discount

        self.grid_size = grid_size

        # number of state
        self.n_state = n_state

        self.memory = Memory(memory_size)

        # Batch size when learning
        self.batch_size = batch_size

    def learned_act(self, s):
        prediction = self.model.predict(np.array([s]))
        expected_rewards = []
        for action in prediction:
            expected_rewards.append(np.dot(action, self.support)/np.sum(action, axis=1))
        return np.argmax(np.array(expected_rewards), axis=0)[0]

    def reinforce(self, s_, n_s_, a_, r_, game_over_, epoch_):
        # Two steps: first memorize the states, second learn from the pool

        self.memory.remember([s_, n_s_, a_, r_, game_over_])

        input_states = np.zeros((self.batch_size, 5,5,self.n_state))
        #target_q = np.zeros((self.batch_size, 4))
        m_prob = [np.ones((self.batch_size, self.num_atoms)) for i in range(4)]

        for i in range(self.batch_size):
            ######## FILL IN
            [s_batch, n_s_batch, a_batch, r_batch, game_over_batch] = self.memory.random_access()
            input_states[i] = s_batch
            #target_q[i] = self.model.predict(np.array([s_batch]))
            #???    m_prob_batch = self.model.predict(np.array([s_batch]))

            z = self.model.predict(np.array([n_s_batch]))
            expected_rewards = []
            for action in z:
                expected_rewards.append(np.dot(action, self.support)/np.sum(action, axis=1))

            optimal_action_idxs = np.argmax(np.array(expected_rewards), axis=0)

            if game_over_:
                ######## FILL IN
                #target_q[i, a_batch] = r_batch

                Tz = min(self.v_max, max(self.v_min, r_batch))
                bj = (Tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[a_batch][i][int(m_l)] += (m_u - bj)
                m_prob[a_batch][i][int(m_u)] += (bj - m_l)
            else:

                ######## FILL IN
                #prediction = self.model.predict(np.array([n_s_batch]))
                #target_q[i, a_batch] = r_batch + self.discount * np.amax(prediction)

                for j in range(self.num_atoms):
                    Tz = min(self.v_max, max(self.v_min, r_batch + self.discount * self.support[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    m_l, m_u = math.floor(bj), math.ceil(bj)

                    m_prob[a_batch][i][int(m_l)] += z[optimal_action_idxs[0]][0][j] * (m_u - bj)


                    m_prob[a_batch][i][int(m_u)] += z[optimal_action_idxs[0]][0][j] * (bj - m_l)
        ######## FILL IN
        # HINT: Clip the target to avoid exploiding gradients.. -- clipping is a bit tighter
        #Clipping the target here?
        l = self.model.train_on_batch(input_states, m_prob)
        # return mean loss? : list of scalars (if the model has multiple outputs and/or metrics).
        return np.mean(l)


    def save(self,name_weights='model.h5',name_model='model.json'):
        self.model.save_weights('models/'+name_weights, overwrite=True)
        with open('models/'+name_model, "w") as outfile:
            json.dump(self.model.to_json(), outfile)

    def load(self,name_weights='model.h5',name_model='model.json'):
        with open('models/'+name_model, "r") as jfile:
            model = model_from_json(json.load(jfile))
        model.load_weights('models'+name_weights)
        model.compile("sgd", "mse")
        self.model = model


class Distributional_DQN_CNN(Distributional_DQN):
    def __init__(self, *args,**kwargs):
        super(Distributional_DQN_CNN, self).__init__(*args,**kwargs)


        state_input = Input(shape=((5,5,self.n_state)))
        cnn_feature = Conv2D(16, (3, 3), strides=(1,1), activation='relu')(state_input)
        #cnn_feature = Conv2D(16, (1, 1), strides=(1,1), activation='relu')(cnn_feature)
        cnn_feature = Flatten()(cnn_feature)
        cnn_feature = Dense(4, activation='relu')(cnn_feature)

        distribution_list = []
        for i in range(4):
            distribution_list.append(Dense(self.num_atoms, activation='softmax')(cnn_feature))

        model = Model(input=state_input, output=distribution_list)

        adam = Adam(lr=0.1)
        model.compile(loss='categorical_crossentropy', optimizer=adam)


        self.model = model
