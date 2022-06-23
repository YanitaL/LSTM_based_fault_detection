"""
In this project, a two layer LSTM model is used to
detect fault of reactors. Pressure singles with
obtained labels (0-1 label) are fed into the
training process in a supervised way.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib as mpl
import matplotlib.pyplot as plt
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
import math
import time
import glob
import random
import sys
import pandas as pd

from my_model import my_model

# Hyperparameters
EPOCHS = 1000000
SUB_EPOCHS = 1
base_learning_rate = 1e-5

tf.keras.backend.set_floatx('float32')
# Batch size
BATCH_SIZE = 128
BATCH_SIZE_VAL = 128
SEQ_LENGTH = 256

use_saved_data = False
DATA_SAMPLING_SLIDE_WINDOWS = 1
use_buffered_state = True
use_trained_weights = True
do_validate = True
not_training = True
if not_training:
    BATCH_SIZE_VAL = 1
do_validate_every = 1
do_test = True
do_test_every = 10
# Random sample the output bit
do_random_sample_output = False
training_fraction = 0.2

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000
RNN_STATE_SIZE = 64
RNN_STATE_SIZE_2 = 64
RNN_STATE_SIZE_3 = 64
DENSE_LAYER_SIZE = 64
dropout_rate_lstm = 0.7
dropout_rate_dense = 0.5
PLOT = 1

# Setting for reading the csv files
n_headers = 0
n_predicted_columns = 1
# n_sensors are the number of channels used for the prediction
n_sensors = 6
# The first column index of data in csv file
data_start_index = 0

NMOVINGAVERAGE = 0


def find_model_layer_index(model, name):
    for idx, layer in enumerate(model.layers):
        if layer.name == name:
            return idx
    return -1

# loading training set
# Should type in the namelist of the training data here
namelist = [
]
random.shuffle(namelist)
numOfFiles = len(namelist)
numOfTrainFiles = int(numOfFiles*0.8)
numOfValFiles = int((numOfFiles-numOfTrainFiles)*0.3)
numOfTestFiles = numOfFiles - numOfTrainFiles - numOfValFiles

train_namelist = namelist[0:numOfTrainFiles]
val_namelist = namelist[numOfTrainFiles:]

# load test data
# Should type in the namelist of the test data
test_namelist = {
}

def read_from_files(namelist, seq_length, DATA_SAMPLING_SLIDE_WINDOWS, average_size=1, off_set=0):
    """
    The csv data has several columns of signals (x) and 1 column of label (y)
    """
    sample_train = []
    sample_label = []
    sample_index = []
    sample_realvalue = []
    start_index = 0
    for i, file in enumerate(namelist):
        _A1 = read_data(file)
        _A1 = _A1[0:(_A1.shape[0]//seq_length)*seq_length]
        _A1 = np.vstack(
            (_A1, np.ones((seq_length, _A1.shape[1]))))
        _A1 = np.float32(_A1)
        __size = -1
        # cutoff the tail sequence
        for i in range((_A1.shape[0])//seq_length-2):
            # All channels are used as input
            slide_count = seq_length//DATA_SAMPLING_SLIDE_WINDOWS
            if use_buffered_state:
                slide_count = 1

            for ii in range(slide_count):
                signal = np.array(
                    _A1[i*seq_length+ii*DATA_SAMPLING_SLIDE_WINDOWS:(i+1)*seq_length+ii*DATA_SAMPLING_SLIDE_WINDOWS, :])
                # Do normalization for each channel
                input_x = signal[0:seq_length, 0:n_sensors]
                sample_train.append(input_x)
                # Customized
                # Please select the desired output
                temp = signal[0:seq_length, [n_sensors+2]]
                temp_realvalue = signal[0:seq_length, [-1]]

                output_x = temp

                sample_label.append(output_x)
                sample_realvalue.append(temp_realvalue)
                if i > 0:
                    sample_index.append([i+start_index-1, i+start_index])
                else:
                    sample_index.append([-1, i+start_index])
                __size = i
        start_index = __size + 1

    lensample = len(sample_train)

    cutoff_size = BATCH_SIZE*(lensample//BATCH_SIZE)
    sample_train = np.array(sample_train, dtype="float32")
    sample_label = np.array(sample_label, dtype="float32")
    sample_realvalue = np.array(sample_realvalue, dtype="float32")
    sample_index = np.array(sample_index, dtype="int")

    sample_train = sample_train[0:cutoff_size]
    sample_label = sample_label[0:cutoff_size]
    sample_index = sample_index[0:cutoff_size]
    sample_realvalue = sample_realvalue[0:cutoff_size]
    return sample_train, sample_label, sample_index, sample_realvalue

model = my_model()
model.build(input_shape=[BATCH_SIZE, SEQ_LENGTH, n_sensors])
model.summary()

def zero_state_array(length, size):
    return np.zeros((length+1, size))

def evaluate_on_dataset(model, train_dataset, train_state_set):
    loss_train_total = 0
    size_total = 0
    prediction_true_value_pair = []
    idx1 = find_model_layer_index(model, "lstm1")
    idx2 = find_model_layer_index(model, "lstm2")
    accuracy = 0
    result = {}
    count = 0
    for train_x, train_y, train_i, _ in train_dataset:
        # First, get the state list for each samples in the batch
        if use_buffered_state:
            state_h_1 = tf.nn.embedding_lookup(
                train_state_set[0], train_i[:, 0]+1)
            state_c_1 = tf.nn.embedding_lookup(
                train_state_set[1], train_i[:, 0]+1)
            state_h_2 = tf.nn.embedding_lookup(
                train_state_set[2], train_i[:, 0]+1)
            state_c_2 = tf.nn.embedding_lookup(
                train_state_set[3], train_i[:, 0]+1)
            init_state = [(state_h_1, state_c_1), (state_h_2, state_c_2)]

        if use_buffered_state:
            model.layers[idx1].reset_states(states=init_state[0])
            model.layers[idx2].reset_states(states=init_state[1])
        else:
            model.layers[idx1].reset_states()
            model.layers[idx2].reset_states()

        predict, state_h_1, state_c_1, state_h_2, state_c_2 = model(
            train_x, training=False)
        if count == 0:
            result = predict
            count = 1
        else:
            result = np.vstack((result, predict))

        loss = tf.keras.losses.binary_crossentropy(
            train_y, predict, from_logits=False)
        loss = tf.reduce_mean(loss)
        b = tf.cast(tf.greater(predict, 0.5), tf.float32)
        accuracy_sample = tf.reduce_sum(b*train_y + (1-b)*(1-train_y))

        samples = train_y.shape[0]
        if use_buffered_state:
            train_state_set[0] = tf.tensor_scatter_nd_update(
                train_state_set[0], train_i[:, 1:2]+1, state_h_1)
            train_state_set[1] = tf.tensor_scatter_nd_update(
                train_state_set[1], train_i[:, 1:2]+1, state_c_1)
            train_state_set[2] = tf.tensor_scatter_nd_update(
                train_state_set[2], train_i[:, 1:2]+1, state_h_2)
            train_state_set[3] = tf.tensor_scatter_nd_update(
                train_state_set[3], train_i[:, 1:2]+1, state_c_2)
        loss_train_total += loss*BATCH_SIZE*SEQ_LENGTH*n_predicted_columns
        size_total += BATCH_SIZE*SEQ_LENGTH*n_predicted_columns
        accuracy += accuracy_sample
    return loss_train_total/size_total, train_state_set, prediction_true_value_pair, accuracy/size_total, result


def train_on_dataset(model, train_dataset, train_state_set, optimizer):
    loss_train_total = 0
    size_total = 0
    prediction_true_value_pair = []
    idx1 = find_model_layer_index(model, "lstm1")
    idx2 = find_model_layer_index(model, "lstm2")
    accuracy = 0
    for train_x, train_y, train_i, _ in train_dataset:
        # First, get the state list for each samples in the batch
        if use_buffered_state:
            state_h_1 = tf.nn.embedding_lookup(
                train_state_set[0], train_i[:, 0]+1)
            state_c_1 = tf.nn.embedding_lookup(
                train_state_set[1], train_i[:, 0]+1)
            state_h_2 = tf.nn.embedding_lookup(
                train_state_set[2], train_i[:, 0]+1)
            state_c_2 = tf.nn.embedding_lookup(
                train_state_set[3], train_i[:, 0]+1)
            init_state = [(state_h_1, state_c_1), (state_h_2, state_c_2)]

        if use_buffered_state:
            model.layers[idx1].reset_states(states=init_state[0])
            model.layers[idx2].reset_states(states=init_state[1])
        else:
            model.layers[idx1].reset_states()
            model.layers[idx2].reset_states()

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)

            predict, state_h_1, state_c_1, state_h_2, state_c_2 = model(
                train_x, training=False)

            # Calculate loss and accuracy
            if do_random_sample_output:
                a = tf.random.uniform(
                    shape=(BATCH_SIZE, SEQ_LENGTH, 1), minval=0, maxval=1, dtype=tf.dtypes.float32)
                a = tf.cast(tf.greater(a, 1-traning_fraction), tf.float32)
                a = tf.repeat(a, repeats=n_predicted_columns, axis=2)
                total = tf.math.reduce_sum(a)
                loss = tf.keras.losses.binary_crossentropy(
                    train_y*a,  predict*a,  from_logits=False)
                b = tf.cast(tf.greater(predict, 0.5), tf.float32)
                accuracy_sample = tf.reduce_sum(
                    b*train_y*a + a*(1-b)*(1-train_y))
                loss = tf.reduce_mean(loss)
                loss = loss*SEQ_LENGTH*BATCH_SIZE*n_predicted_columns/total
            else:
                loss = tf.keras.losses.binary_crossentropy(
                    train_y, predict, from_logits=False)
                loss = tf.reduce_mean(loss)
                b = tf.cast(tf.greater(predict, 0.5), tf.float32)
                accuracy_sample = tf.reduce_sum(b*train_y + (1-b)*(1-train_y))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        samples = train_y.shape[0]
        if use_buffered_state:
            train_state_set[0] = tf.tensor_scatter_nd_update(
                train_state_set[0], train_i[:, 1:2]+1, state_h_1)
            train_state_set[1] = tf.tensor_scatter_nd_update(
                train_state_set[1], train_i[:, 1:2]+1, state_c_1)
            train_state_set[2] = tf.tensor_scatter_nd_update(
                train_state_set[2], train_i[:, 1:2]+1, state_h_2)
            train_state_set[3] = tf.tensor_scatter_nd_update(
                train_state_set[3], train_i[:, 1:2]+1, state_c_2)
        if do_random_sample_output:
            loss_train_total += loss*total
            size_total += total
            accuracy += accuracy_sample
        else:
            loss_train_total += loss*BATCH_SIZE*SEQ_LENGTH*n_predicted_columns
            size_total += BATCH_SIZE*SEQ_LENGTH*n_predicted_columns
            accuracy += accuracy_sample
    return loss_train_total/size_total, train_state_set, prediction_true_value_pair, accuracy/size_total

if not use_saved_data:
    train_data, train_label, train_index, train_realvalue = read_from_files(
        train_namelist, seq_length=SEQ_LENGTH)
    val_data, val_label, val_index, val_realvalue = read_from_files(
        val_namelist, seq_length=SEQ_LENGTH)
    test_data, test_label, test_index, test_realvalue = read_from_files(
        test_namelist, seq_length=SEQ_LENGTH)
    np.savez("data_set_sf.npz",
             train_data=train_data, train_label=train_label, train_index=train_index, train_realvalue=train_realvalue,
             val_data=val_data, val_label=val_label, val_index=val_index, val_realvalue=val_realvalue,
             test_data=test_data, test_label=test_label, test_index=test_index, test_realvalue=test_realvalue
             )
else:
    data_set = np.load("quant_data_set_sf.npz")
    train_data = data_set["train_data"]
    train_label = data_set["train_label"]
    train_index = data_set["train_index"]
    train_realvalue = data_set["train_realvalue"]
    val_data = data_set["val_data"]
    val_label = data_set["val_label"]
    val_index = data_set["val_index"]
    val_realvalue = data_set["val_realvalue"]
    test_data = data_set["test_data"]
    test_label = data_set["test_label"]
    test_index = data_set["test_index"]
    test_realvalue = data_set["test_realvalue"]
with open("convergence_history_prediction.dat", "w") as history_file_handle:
    # Use pretrained weights to train the model
    if use_trained_weights:
        model.load_weights("./prediction_sf_best_acc_0_6.h5")

    model_val = my_model()
    model_val.build(input_shape=[BATCH_SIZE_VAL, SEQ_LENGTH, n_sensors])

    model_test = my_model()
    model_test.build(input_shape=[1, SEQ_LENGTH, n_sensors])

    optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate)
    model.compile(optimizer=optimizer,
                  loss={"main_output": "binary_crossentropy"}
                  )
    model_val.compile(optimizer=optimizer,
                      loss={"main_output": "binary_crossentropy"}
                      )
    lowest_loss = 1e10

    train_state_set_h_1 = zero_state_array(
        train_data.shape[0], RNN_STATE_SIZE)
    train_state_set_c_1 = zero_state_array(
        train_data.shape[0], RNN_STATE_SIZE)
    train_state_set_h_2 = zero_state_array(
        train_data.shape[0], RNN_STATE_SIZE_2)
    train_state_set_c_2 = zero_state_array(
        train_data.shape[0], RNN_STATE_SIZE_2)

    # Store c-state and h-state for training set
    train_state_set = {}
    train_state_set[0] = train_state_set_c_1
    train_state_set[1] = train_state_set_h_1
    train_state_set[2] = train_state_set_c_2
    train_state_set[3] = train_state_set_h_2

    val_state_set_h_1 = zero_state_array(
        val_data.shape[0], RNN_STATE_SIZE)
    val_state_set_c_1 = zero_state_array(
        val_data.shape[0], RNN_STATE_SIZE)
    val_state_set_h_2 = zero_state_array(
        val_data.shape[0], RNN_STATE_SIZE_2)
    val_state_set_c_2 = zero_state_array(
        val_data.shape[0], RNN_STATE_SIZE_2)

    # Store c-state and h-state for validation set
    val_state_set = {}
    val_state_set[0] = val_state_set_c_1
    val_state_set[1] = val_state_set_h_1
    val_state_set[2] = val_state_set_c_2
    val_state_set[3] = val_state_set_h_2

    if not_training:
        EPOCHS = 1
    for epoch in range(EPOCHS):
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label, train_index, train_realvalue)).shuffle(
            BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (val_data, val_label, val_index, val_realvalue)).batch(BATCH_SIZE_VAL, drop_remainder=True)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_data, test_label, test_index, test_realvalue)).batch(1, drop_remainder=True)
        for sub_epoch in range(SUB_EPOCHS):
            start_time = time.time()
            loss_train_total = 0
            size_total = 0
            count = 0
            loss_train_dataset = 0
            accuracy = 0
            if not not_training:
                loss_train_dataset, train_state_set, _, accuracy = train_on_dataset(
                    model, train_dataset, train_state_set, optimizer)
                model_name = 'prediction_sf_tmp.h5'
                model.save_weights(model_name)

            val_loss = 0
            accuracy_val = 0
            if not not_training:
                if do_validate and (epoch*SUB_EPOCHS+sub_epoch) % do_validate_every == 0:
                    model_val.load_weights(model_name)
                    model_val.reset_states()

                    val_loss, val_state_set, _, accuracy_val, predict_y = evaluate_on_dataset(
                        model_val, val_dataset, val_state_set)

                    model_name = 'prediction_sf_best.h5'
                    if val_loss < lowest_loss and not not_training:
                        model.save_weights(model_name)
                        lowest_loss = val_loss

            end_time = time.time()
            do_test_this_time = do_test and (
                epoch*SUB_EPOCHS+sub_epoch) % do_test_every == 0
            if not do_test_this_time:
                print('Epoch: {:6d}, sub: {:6d}'
                      '   Train set loss: {:12.8f}'
                      '   Train set accu: {:12.8f}'
                      '   Val set loss: {:12.8f}'
                      '   Val set accu: {:12.8f}'
                      '   time elapsed for {:10.3f} seconds'.format(
                          epoch, sub_epoch,
                          loss_train_dataset,
                          accuracy,
                          val_loss,
                          accuracy_val,
                          end_time - start_time))

            test_loss = 0
            accuracy_test = 0
            if do_test_this_time:
                model_name = 'prediction_sf_best_acc_0_6.h5'
                model_test.load_weights(model_name)
                model_test.reset_states()
                predict_y, ss1, ss2, ss3, ss4 = model_test.predict(
                    test_data, batch_size=1, verbose=0)
                diff = tf.keras.losses.binary_crossentropy(
                    predict_y, test_label)
                a = tf.cast(tf.greater(predict_y, 0.5), tf.float32)
                accuracy_test = tf.reduce_sum(a*test_label + (1-a)*(1-test_label))/(
                    test_label.shape[0]*test_label.shape[1]*test_label.shape[2])
                test_loss = tf.reduce_mean(diff)

                print('Epoch: {:6d}, sub: {:6d}'
                      '   Train set loss: {:12.8f}'
                      '   Train set accu: {:12.8f}'
                      '   Val set loss: {:12.8f}'
                      '   Val set accu: {:12.8f}'
                      '   Tes set loss: {:12.8f}'
                      '   Tes set accu: {:12.8f}'
                      '   time elapsed for {:10.3f} seconds'.format(
                          epoch, sub_epoch,
                          loss_train_dataset,
                          accuracy,
                          val_loss,
                          accuracy_val,
                          test_loss,
                          accuracy_test,
                          end_time - start_time))

            sys.stdout.flush()
            history_file_handle.write(
                str(epoch*SUB_EPOCHS+sub_epoch)+" "+str(epoch) + " " + str(sub_epoch))
            history_file_handle.write(
                " " + str(float(loss_train_dataset)))
            history_file_handle.write(
                " " + str(float(accuracy)))
            history_file_handle.write(
                " " + str(float(val_loss)))
            history_file_handle.write(
                " " + str(float(accuracy_val)))
            history_file_handle.write(
                " " + str(float(test_loss)))
            history_file_handle.write(
                " " + str(float(accuracy_test)))
            history_file_handle.write("\n")
            history_file_handle.flush()
history_file_handle.close()
