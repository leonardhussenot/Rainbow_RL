import keras
import numpy as np
import io
import base64
from IPython.display import HTML
import skvideo.io
import cv2
import json
import os

from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Flatten
from keras.optimizers import sgd
from keras.layers import Conv2D, MaxPooling2D, Activation, AveragePooling2D,Reshape,BatchNormalization

from Agent import *
from Memory import *

class Multistep_DQN(Agent):
    def __init__(self, grid_size, n = 3, discount = 0.9, epsilon = 0.1, memory_size=100, batch_size = 16, n_state=3):
        super(Multistep_DQN, self).__init__(epsilon = epsilon)

        # n-step TD
        self.n = n

        # Discount for Q learning
        self.discount = discount

        self.grid_size = grid_size

        # number of state
        self.n_state = n_state

        self.memory = Memory(memory_size)

        # Batch size when learning
        self.batch_size = batch_size

        self.time_step = 0

    def learned_act(self, s):
        prediction = self.model.predict(np.array([s,]))
        return np.argmax(prediction)

    def reinforce(self, s_, n_s_, a_, r_, game_over_, epoch_):
        if self.time_step == 0:
            self.current_r_sum = 0
            self.first_state = s_
            self.first_action = a_

        self.current_r_sum += r_
        self.time_step += 1

        if self.time_step == self.n or game_over_:

            self.multistep_reinforce(self.first_state,
                                n_s_, self.first_action,
                                self.current_r_sum, game_over_,
                                epoch_, time_step=self.time_step)
            self.time_step = 0


    def multistep_reinforce(self, s_, n_s_, a_, r_, game_over_, epoch_, time_step):

        # Two steps: first memorize the states, second learn from the pool
        self.memory.remember([s_, n_s_, a_, r_, game_over_])

        input_states = np.zeros((self.batch_size, 5,5,self.n_state))
        target_q = np.zeros((self.batch_size, 4))

        for i in range(self.batch_size):
            ######## FILL IN
            [s_batch, n_s_batch, a_batch, r_batch, game_over_batch] = self.memory.random_access()
            input_states[i] = s_batch
            target_q[i] = self.model.predict(np.array([s_batch]))
            if game_over_:
                ######## FILL IN
                target_q[i, a_batch] = r_batch
            else:
                ######## FILL IN
                prediction = self.model.predict(np.array([n_s_batch]))
                target_q[i, a_batch] = r_batch + (self.discount ** time_step) * np.amax(prediction)
        ######## FILL IN
        # HINT: Clip the target to avoid exploiding gradients.. -- clipping is a bit tighter
        target_q = np.clip(target_q, -3, 3)
        l = self.model.train_on_batch(input_states, target_q)
        return l

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


class Multistep_DQN_CNN(Multistep_DQN):
    def __init__(self, *args,**kwargs):
        super(Multistep_DQN_CNN, self).__init__(*args,**kwargs)

        ###### FILL IN
        model = Sequential()

        model.add(Conv2D(32, (1,1),strides=(1,1),input_shape=(5,5,self.n_state)))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (1,1),strides=(1,1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(4))
        model.add(Activation('tanh'))


        model.compile(sgd(lr=0.1, decay=1e-4, momentum=0.0), "mse")
        #model.compile(Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), "mse")
        self.model = model
