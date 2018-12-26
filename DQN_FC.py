class DQN_FC(DQN):
def __init__(self,*args,**kwargs):
super(DQN_FC, self).__init__( *args,**kwargs)

# NN Model

####### FILL IN
model = Sequential()

model.add(Flatten(input_shape=(5,5,2)))
model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(4))
model.add(Activation('relu'))

model.compile(sgd(lr=0.1, decay=1e-4, momentum=0.0), "mse")
self.model = model
