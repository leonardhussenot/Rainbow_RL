import keras
import numpy as np
import io
import base64
from IPython.display import HTML
import skvideo.io
import cv2
import json
import os

from keras.models import Sequential,model_from_json, Model
from keras.layers.core import Dense, Flatten
from keras.optimizers import sgd, Adam
from keras.layers import Lambda, Input, Conv2D, MaxPooling2D, Activation, AveragePooling2D,Reshape,BatchNormalization
from keras import backend as K

from Agent import *
from Memory import *




class Dueling_DQN(Agent):
    def __init__(self, grid_size,  epsilon = 0.1, memory_size=100, batch_size = 16, n_state=3):
        super(Dueling_DQN, self).__init__(epsilon = epsilon)

        # Discount for Q learning
        self.discount = 0.9

        self.grid_size = grid_size

        # number of state
        self.n_state = n_state

        self.memory = Memory(memory_size)

        # Batch size when learning
        self.batch_size = batch_size



    def dueling_loss(self,l):
        value = l[0]
        advantage = l[1]
        return value + (advantage - K.mean(advantage, axis=1, keepdims=True))

    def createLayers(self):
        x = Input(shape=(5,5,3))

        conv1 = Activation('relu')(Conv2D(16, (3,3),strides=(1,1),input_shape=(5,5,self.n_state))(x))
        #conv2 = Activation('relu')(Conv2D(16, (1,1),strides=(1,1))(conv1))
        f = Flatten()(conv1)
        y = Activation('tanh')(Dense(5)(f))

        ## Calculating V(s)
        value_fc = Activation('relu')(Dense(5)(y))
        value = Dense(1)(value_fc)

        ## Calculating A(s,a)
        advantage_fc = Activation('relu')(Dense(5)(y))
        advantage = Dense(4)(advantage_fc)

        z = Lambda(self.dueling_loss, output_shape=(4,))([value, advantage])
        #z = Lambda(lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
        #               output_shape=(4,)) (y)
        return x, z


    def learned_act(self, s):
        prediction=self.model.predict(np.array([s,]))
        return np.argmax(prediction)

    def reinforce(self, s_, n_s_, a_, r_, game_over_, epoch_):
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
                target_q[i, a_batch] = r_batch + self.discount * np.amax(prediction)
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

class Dueling_DQN_CNN(Dueling_DQN):
    def __init__(self, *args,**kwargs):
        super(Dueling_DQN_CNN, self).__init__(*args,**kwargs)

        x, z = self.createLayers()
        model = Model(input=x, output=z)


        model.compile(sgd(lr=0.1, decay=1e-4, momentum=0.0), "mse")
        #model.compile(Adam(lr=0.5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), "mse")
        self.model = model
