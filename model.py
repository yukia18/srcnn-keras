import os
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from keras.layers.core import Activation

class SRCNN:
    def __init__(self, image_size, label_size, c_dim, learning_rate, batch_size, epochs, is_training):
        self.image_size = image_size
        self.label_size = label_size
        self.c_dim = c_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.is_training = is_training
        if self.is_training:
            self.model = self.build_model()
        else:
            self.model = self.load()    
    
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(64,9,input_shape=(self.image_size,self.image_size,self.c_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(32,1))
        model.add(Activation('relu'))
        model.add(Conv2D(self.c_dim,5))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
        return model
    
    def train(self, X_train, Y_train):
        history = self.model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1)
        if self.is_training:
            self.save()
        return history
    
    def process(input):
        predicted = self.model.predict(input)
        return predicted
    
    def load(self):
        '''
        model_filename = 'srcnn_model.json'
        json_string = open(os.path.join('./',model_filename)).read()
        model = model_from_json(json_string)
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
        '''
        weight_filename = 'srcnn_model_weight.hdf5'
        model = self.build_model()
        model.load_weights(os.path.join('./',weight_filename))
        return model

    def save(self):
        json_string = self.model.to_json()
        open(os.path.join('./','srcnn_model.json'),'w').write(json_string)
        self.model.save_weights(os.path.join('./','srcnn_model_weight.hdf5'))
        return json_string
