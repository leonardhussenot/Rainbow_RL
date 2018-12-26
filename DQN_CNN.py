class DQN_CNN(DQN):
    def __init__(self, *args,**kwargs):
        super(DQN_CNN, self).__init__(*args,**kwargs)

        ###### FILL IN
        model = Sequential()

        model.add(Conv2D(32, (1,1),strides=(1,1),input_shape=(5,5,self.n_state)))
        model.add(Conv2D(16, (1,1),strides=(1,1)))
        model.add(Flatten())
        model.add(Dense(4))
        model.add(Activation('relu'))


        model.compile(sgd(lr=0.1, decay=1e-4, momentum=0.0), "mse")
        self.model = model