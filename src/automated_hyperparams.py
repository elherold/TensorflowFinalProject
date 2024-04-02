from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

import pandas as pd

import keras_tuner as kt
import numpy as np

from merge_train_test import getting_datasets
from model_testrun import create_embedding_matrix, tokenizer_padding

import os
print("imports done")

class ModifyInputLength(tf.keras.callbacks.Callback):
  def __init__(self, input_arrays):
    self.input_arrays = input_arrays
    super(ModifyInputLength, self).__init__()

  def on_epoch_begin(self, epoch, logs=None):
    # Get the current hyperparameter value for input_length
    current_length = self.model.hyperparameters.get('input_length')
    # Access the corresponding input array from the dictionary
    self.model.x = self.input_arrays[current_length]

def model_builder(hp):
    # Create the model
    model = Sequential()
    hp_input_length = hp.Choice('input_length', values=[100, 200, 500])
    model.add(Embedding(input_dim=len(word_index) + 1, output_dim=100, input_length=hp_input_length, weights=[embedding_matrix], trainable=False)) # set trainable to False to keep the embeddings fixed
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 16-128
    hp_lstm_units = hp.Int('lstm_units', min_value=16, max_value=128, step=32)
    model.add(LSTM(hp_lstm_units))
    # Tune the dropout rate in the Dropout layer
    # Choose an optimal value from 0.1, 0.2, 0.3, 0.4, or 0.5
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(hp_dropout))
    model.add(Dense(1, activation='sigmoid'))
    
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), 
                loss='binary_crossentropy', metrics=['accuracy'])
    return model

train_sets, X_test, y_test = getting_datasets()
X_train = train_sets['original_data'][0][0:2000]
y_train = train_sets['original_data'][1][0:2000]

# Tokenize and pad the training and testing sequences
X_train = list(X_train)


X_train_padded, X_test_padded, tokenizer, word_index, embedding_matrix = tokenizer_padding(X_train, X_test, "original_data", [100, 200, 500])

print(X_train_padded[0].shape)
print(X_train[0])
print(len(X_train[0]))
print(X_train_padded[0])
y_train = np.array(y_train, dtype=np.int32)
y_test = np.array(y_test, dtype=np.int32)

# Before tuning, create multiple versions of your input arrays with lengths 100, 200, and 500. You can achieve this using padding or truncation techniques.
input_arrays = {
    100: X_train_padded[0],
    200: X_train_padded[1],
    500: X_train_padded[2]
}
# Create the custom callback instance
modify_input_length_callback = ModifyInputLength(input_arrays)

tuner = kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=10, factor=3, directory='my_dir', project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

tuner.search(X_train_padded, y_train, epochs=50, validation_split=0.2, callbacks=[modify_input_length_callback, stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]



print(f"""
The hyperparameter search is complete. The optimal input_length is {best_hps.get('input_length')},
the optimal number of LSTM units is {best_hps.get('lstm_units')},
the optimal dropout rate is {best_hps.get('dropout')},
and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")


print(f"""Hyperparameter object:
{best_hps}
""")

best_hps_list=tuner.get_best_hyperparameters(num_trials=5)
print("input_length, lstm_units, dropout, learning_rate")
print(best_hps_list.get('input_length'))
print(best_hps_list.get('lstm_units'))
print(best_hps_list.get('dropout'))
print(best_hps_list.get('learning_rate'))
