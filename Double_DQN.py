from Environment import *
from Memory import *
from Agent import *

import json
import numpy as np
from keras.models import Sequential,model_from_json, clone_model
from keras.layers.core import Dense, Flatten
from keras.optimizers import sgd, Adam
from keras.layers import Conv2D, MaxPooling2D, Activation, AveragePooling2D,Reshape,BatchNormalization

class Double_DQN(Agent):
    def __init__(self, grid_size, discount=0.9, epsilon=0.1, memory_size=100, batch_size=16, n_state=3):
        super(Double_DQN, self).__init__(epsilon = epsilon)

        # Discount for Q learning
        self.discount = discount

        self.grid_size = grid_size

        # number of state
        self.n_state = n_state

        # Memory
        self.memory = Memory(memory_size)

        # Batch size when learning
        self.batch_size = batch_size

    def learned_act(self, s):
        ## TODO : vérifier que c'est bien ça mais le prof l'avait dit il me semble
        prediction = self.model.predict(np.array([s,]))
        return np.argmax(prediction)

    def reinforce(self, s_, n_s_, a_, r_, game_over_, epoch_):
        # Two steps: first memorize the states, second learn from the pool

        self.memory.remember([s_, n_s_, a_, r_, game_over_])

        input_states = np.zeros((self.batch_size, 5,5,self.n_state))
        target_q = np.zeros((self.batch_size, 4))

        ## PROPOSITION CHANGEMENT LÉONARD
        #if (epoch_ % 3 == 0) : # La fréquence de mise à jour des poids est un hyper paramètre
        #    self.target_model.set_weights(self.model.get_weights())
        zero_one = np.random.randint(2)

        for i in range(self.batch_size):
            ######## FILL IN
            [s_batch, n_s_batch, a_batch, r_batch, game_over_batch] = self.memory.random_access()
            input_states[i] = s_batch
            
            if zero_one == 0:
                target_q[i] = self.model.predict(np.array([s_batch]))
            else :
                target_q[i] = self.model2.predict(np.array([s_batch]))
                
            if game_over_:
                ######## FILL IN
                target_q[i, a_batch] = r_batch
            else:
                ######## FILL IN
                ## PROPOSITION CHANGEMENT

                if zero_one == 0:
                    prediction1 = self.model.predict(np.array([n_s_batch]))
                    prediction2 = self.model2.predict(np.array([n_s_batch]))
                else:
                    prediction2 = self.model.predict(np.array([n_s_batch]))
                    prediction1 = self.model2.predict(np.array([n_s_batch]))


                target_q[i, a_batch] = r_batch + self.discount * (prediction2.ravel())[np.argmax(prediction1)]
        ######## FILL IN
        # HINT: Clip the target to avoid exploiding gradients.. -- clipping is a bit tighter
        target_q = np.clip(target_q, -3, 3)
        if zero_one == 0:
            l = self.model.train_on_batch(input_states, target_q)
        else:
            l = self.model2.train_on_batch(input_states, target_q)



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


class Double_DQN_CNN(Double_DQN):
    def __init__(self, *args,**kwargs):
        super(Double_DQN_CNN, self).__init__(*args,**kwargs)

        ###### FILL IN
        model = Sequential()

        model.add(Conv2D(16, (3,3),strides=(1,1),input_shape=(5,5,self.n_state)))
        model.add(Activation('relu'))
        #model.add(Conv2D(16, (1,1),strides=(1,1)))
        #model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(4))
        model.add(Activation('tanh'))


        model.compile(sgd(lr=0.1, decay=1e-4, momentum=0.0), "mse")
        self.model = model
        self.model2 = clone_model(self.model)
        self.model2.compile(sgd(lr=0.1, decay=1e-4, momentum=0.0), "mse")
        self.model2.set_weights(self.model.get_weights())
