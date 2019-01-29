import keras
import numpy as np
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Flatten
from keras.optimizers import sgd
from keras.layers import Conv2D, MaxPooling2D, Activation, AveragePooling2D,Reshape,BatchNormalization

from DQN import DQN
from Memory import *

class DQN_CNN(DQN):
    def __init__(self, *args,**kwargs):
        super(DQN_CNN, self).__init__(*args,**kwargs)

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
        #model.compile(Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), "mse")
        self.model = model
