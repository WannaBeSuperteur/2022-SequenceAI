import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import SimpleRNN

# load input and output data
def loadData(name):

    # load the input and output
    input_ = np.array(pd.read_csv(name + '_input_converted.csv'))
    input_ = input_[:, 1:]
    
    output_ = np.array(pd.read_csv(name + '_output_converted.csv'))
    output_ = output_[:, 1:]

    # reshape the input
    input_ = np.reshape(input_, (-1, 15, 9))

    return (input_, output_)

# define RNN model using LSTM layer
def defineRNNModel():

    # L2 regularization
    L2 = tf.keras.regularizers.l2(0.001)

    # define the model
    model = tf.keras.Sequential()
    model.add(layers.LSTM(256, input_shape=(15, 9)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(9))

    # compile the model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

if __name__ == '__main__':

    # load training input and output data
    (train_input, train_output) = loadData('train')

    # define the RNN model
    model = defineRNNModel()
    
    # train the model
    model.fit(train_input, train_output, validation_split=0.05, epochs=5)
    model.save('RNN_model')

    # load test input and output data
    (test_input, test_output) = loadData('test')

    # predict about the test input and save the prediction
    prediction = model.predict(test_input)
    pd.DataFrame(prediction).to_csv('test_prediction.csv')
