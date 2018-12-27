from Environment import *
from Memory import *
from Agent import *

import json
import numpy as np
from keras.models import Sequential,model_from_json, clone_model
from keras.layers.core import Dense, Flatten
from keras.optimizers import sgd
from keras.layers import Conv2D, MaxPooling2D, Activation, AveragePooling2D,Reshape,BatchNormalization

class Prioritized_DQN(Agent):
    def __init__(self, grid_size, discount = 0.9, epsilon = 0.1, memory_size=100, batch_size = 16,n_state=3):
        super(Prioritized_DQN, self).__init__(epsilon = epsilon)

        # Discount for Q learning
        self.discount = discount

        self.grid_size = grid_size

        self.memory_size = memory_size

        # Memory
        self.memory = Memory_prioritized(self.memory_size)

        # number of state
        self.n_state = n_state

        # Batch size when learning
        self.batch_size = batch_size

    def learned_act(self, s):
        prediction = self.model.predict(np.array([s,]))
        return np.argmax(prediction)

    def reinforce(self, s_, n_s_, a_, r_, game_over_, epoch_):
        # Two steps: first memorize the states, second learn from the pool

        self.memory.prioritised_remember([s_, n_s_, a_, r_, game_over_], self.model, self.discount)

        input_states = np.zeros((self.batch_size, 5,5,self.n_state))
        target_q = np.zeros((self.batch_size, 4))

        for i in range(self.batch_size):
            ######## FILL IN
            [s_batch, n_s_batch, a_batch, r_batch, game_over_batch] = self.memory.prioritized_access()
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



class Prioritized_DQN_CNN(Prioritized_DQN):
    def __init__(self, *args,**kwargs):
        super(Prioritized_DQN_CNN, self).__init__(*args,**kwargs)

        ###### FILL IN
        model = Sequential()

        model.add(Conv2D(32, (1,1),strides=(1,1),input_shape=(5,5,self.n_state)))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (1,1),strides=(1,1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(4))
        model.add(Activation('tanh'))


        model.compile(sgd(lr=0.01, decay=1e-4, momentum=0.0), "mse")
        self.model = model
