import keras
import numpy as np
from keras.models import Sequential,model_from_json, Model
from keras.layers.core import Dense, Flatten
from keras.optimizers import sgd
from keras.layers import Conv2D, Input, MaxPooling2D, Activation, AveragePooling2D,Reshape,BatchNormalization, GaussianNoise, Multiply, Add, Subtract
from keras import backend as K

from DQN import DQN
from Memory import *

def createLayers(depth):
    x = Input(shape=(5,5,depth))
    
    conv1 = Activation('relu')(Conv2D(16, (3,3),strides=(1,1),input_shape=(5,5,depth))(x))
    #conv2 = Activation('relu')(Conv2D(16, (1,1),strides=(1,1))(conv1))
    f = Flatten()(conv1)
    y1 = (Dense(4)(f)) 
    zeros = Subtract()([y1,y1])
    noise = GaussianNoise(0.5)((zeros))
    perturbation = Multiply()([Dense(4)(f),noise])
    #perturbation = noise
    #perturbation = zeros
    
    output = Activation('tanh')( Add()([y1,perturbation])) 

    return x, output
    
   
class Noisy_DQN(DQN):
    def __init__(self, *args,**kwargs):
        super(Noisy_DQN, self).__init__(*args,**kwargs)

        x, z = createLayers(self.n_state)
        model = Model(input=x, output=z)


        model.compile(sgd(lr=0.1, decay=1e-4, momentum=0.0), "mse")
        self.model = model