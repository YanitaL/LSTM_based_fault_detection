"""
A multilayer LSTM model to train data with time series
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
import numpy as np
import os

import time
import glob
import random
import sys

class my_model(tf.keras.Model):
    """
    A multilayer LSTM model to train the data
    """
    def __init__(self):
        super(my_model, self).__init__()

        # Initialize the model

        self.lstm_layer_1 = tf.keras.layers.LSTM(RNN_STATE_SIZE,
                                                 return_sequences=True,
                                                 stateful=True,
                                                 return_state=True,
                                                 unroll=False,
                                                 recurrent_initializer='glorot_uniform', name="lstm1")
        self.lstm_layer_2 = tf.keras.layers.LSTM(RNN_STATE_SIZE,
                                                 return_sequences=True,
                                                 stateful=True,
                                                 return_state=True,
                                                 unroll=False,
                                                 recurrent_initializer='glorot_uniform', name="lstm2")
        # //self.lstm_layer_3 = tf.keras.layers.LSTM(RNN_STATE_SIZE,
        # //                                    return_sequences=True,
        # //                                    stateful=True,
        # //                                    return_state=True,
        # //                                    unroll = False,
        # //                                    recurrent_initializer='glorot_uniform', name="lstm3")
        self.dropout_lstm_layer1 = TimeDistributed(Dropout(dropout_rate_lstm))
        self.dropout_lstm_layer2 = TimeDistributed(Dropout(dropout_rate_lstm))
        # //self.dropout_lstm_layer3 =  TimeDistributed(Dropout(dropout_rate_lstm))
        # //self.dropout_layer1 = TimeDistributed(Dropout(dropout_rate_dense))
        # //self.dropout_layer2 = TimeDistributed(Dropout(dropout_rate_dense))
        self.dense_layer1 = Dense(RNN_STATE_SIZE, activation="gelu")
        self.dense_layer2 = Dense(RNN_STATE_SIZE, activation="gelu")
        self.output_layer = Dense(n_predicted_columns,
                                  name="main_output", activation="sigmoid")

    def call(self, inputs, training=False):
        """
        :param inputs: float32 numpy array
        :returns outputs, h states and c states for both LSTM layers
        """

        # two LSTM models
        lstm_1, state_h_1, state_c_1 = self.lstm_layer_1(inputs)
        lstm_1 = self.dropout_lstm_layer1(lstm_1, training=training)
        lstm_2, state_h_2, state_c_2 = self.lstm_layer_2(lstm_1)
        lstm_2 = self.dropout_lstm_layer2(lstm_2, training=training)

        # Add more layers if needed
        # //lstm_3, state_h_3, state_c_3 = self.lstm_layer_3(lstm_2)
        # //lstm_3 = self.dropout_lstm_layer3(lstm_3,training=training)

        # feedforward fully connected layer, Gaussian Error Linear Units as activation function
        x = self.dense_layer1(lstm_2)
        x = self.dense_layer2(x)
        x = x + lstm_2
        # fully connected layer as output layer, Sigmoid as activation function
        main_output = self.output_layer(x)
        # //return main_output, state_h_1, state_c_1, state_h_2, state_c_2, state_h_3, state_c_3
        return main_output, state_h_1, state_c_1, state_h_2, state_c_2