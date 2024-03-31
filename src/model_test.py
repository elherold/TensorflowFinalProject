
# create the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from merge_train_test import getting_datasets
import gensim.downloader as api
from preprocessing_comments import preprocess_text
import json 
import numpy as np

embedding_ap = api.load("glove-wiki-gigaword-100")

def tokenizer_padding(X_train, y_train, X_test, y_test):
    """
    Tokenizes and pads the training and testing text sequences, and converts labels to numpy arrays.

    This function takes the training and testing text data (X_train, X_test), tokenizes these texts into sequences 
    of integers, and then pads these sequences to ensure that they have the same length. Additionally, it converts 
    the label lists (y_train, y_test) into numpy arrays for use in machine learning models.

    Parameters:
    X_train (list of str): The list of training sentences.
    y_train (list): The list of training labels.
    X_test (list of str): The list of testing sentences.
    y_test (list): The list of testing labels.

    Returns:
    tuple: Returns four elements:
        - X_train_padded (numpy.ndarray): Padded sequences for the training set.
        - y_train_array (numpy.ndarray): Array of training labels.
        - X_test_padded (numpy.ndarray): Padded sequences for the testing set.
        - y_test_array (numpy.ndarray): Array of testing labels.

    The padding is done based on the maximum sequence length found in either the training or testing sets.
    """
    # I have a dict saved as a json file, I want to apply my preprocessing function to the values of its keys and then tokenize and pad the sequences 


    # Load the dictionary from the json file
    with open('../data/test_predictions.json', 'r') as file:
        predictions_dict = json.load(file)

    # Apply your preprocessing function to the values of the dictionary
    preprocessed_dict = {}
    for key, value in predictions_dict.items():
        preprocessed_value = preprocess_text(value)
        preprocessed_dict[key] = preprocessed_value

    # Convert the preprocessed values to sequences
    sequences = tokenizer.texts_to_sequences(list(preprocessed_dict.values()))

    # Pad the sequences
    padded_sequences = pad_sequences(sequences, maxlen=500)

    # Convert the dictionary keys to a list
    keys = list(preprocessed_dict.keys())

    # Create a new dictionary with the preprocessed and padded sequences
    predictions_dict = {keys[i]: padded_sequences[i] for i in range(len(keys))}

    # tokenize sentences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train) # fit only training data

    train_sequence = tokenizer.texts_to_sequences(X_train)
    test_sequence = tokenizer.texts_to_sequences(X_test)

    # Pad the sequences
    #max_sequence_length = max(max(len(x) for x in train_sequence), max(len(x) for x in test_sequence))
    #print(f"Max sequence length is: {max_sequence_length}.")
    X_train_padded = pad_sequences(train_sequence, maxlen=500)
    X_test_padded = pad_sequences(test_sequence, maxlen=500)

    # Convert y_train to a numpy array
    y_train_array = np.array(y_train)

    # Same for y_test if it's not already a numpy array
    y_test_array = np.array(y_test)
    
    return X_train_padded, y_train_array, X_test_padded, y_test_array, predictions_dict

train_sets, x_test, y_test = getting_datasets()

dict_datasets = {}
# Process each training set
for name, (X_train, y_train) in train_sets.items():
    # Tokenize and pad the training and testing sequences
    X_train_padded, y_train_array, X_test_padded, y_test_array, predictions_dict = tokenizer_padding(X_train, y_train, X_test, y_test)
    
    # Add the datasets to the dictionary
    dict_datasets[name] = [X_train_padded, y_train_array, X_test_padded, y_test_array]


for key, value in dict_datasets.items():
    print(f"Key: {key}, Value: {value}")

    X_train, X_test, y_train, y_test = value

    model = Sequential()
    model.add(Embedding(input_dim=len(word_index) + 1, output_dim=100, input_length=500))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))  # Dropout zur Reduzierung von Overfitting
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=4, batch_size=32, validation_data=(X_test, y_test))
    history_dict = history.history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'../models/{key}_history.csv')
    model.save('../models/{key}_model')
    print(model.summary())


