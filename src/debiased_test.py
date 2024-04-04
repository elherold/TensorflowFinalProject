from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
import numpy as np
import json
import pickle
from merge_train_test import getting_datasets
import os
from datetime import datetime
import tensorflow as tf


def create_embedding_matrix(word_index, embedding_dim=100):
    """
    Creates an embedding matrix for the Embedding Layer from the GloVe embeddings.

    Parameters:
    word_index(dict): A dictionary mapping words to their index in the Tokenizer. 
    embedding_dimension (int): The dimension of the embedding layer.

    Returns: test change
    numpy.ndarray: An embedding matrix where the ith row gives the embedding of the word with index i.
    """
    # Load the GloVe embeddings
    embeddings_index = {}
    with open('../data/glove.6B.100d.txt', encoding='utf-8') as f:
        # Fill up the embedding matrix dictionary with the natural language words as keys and their respective vector representations as values
        print(type(word_index))
        l = 0
        for line in f:
            l += 1
            if l % 1000 == 0:
                print(f"Processed {l} lines")
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    # Prepare embedding matrix
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if i % 1000 == 0:
            print(f"Processed {i} words")
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def tokenizer_padding(X_train, X_test, name, max_length=[500]):
    """
    Tokenizes and pads the training and testing text sequences.
    Loads the tokenizer and word_index if they exist, otherwise creates them.
    """
    
    try:
        embedding_matrix = np.load(f'../models/debiased_embedding_matrix_{name}.npy')
        word_index = json.load(open(f'../models/word_index_{name}.json'))
        tokenizer = pickle.load(open(f'../models/tokenizer_{name}.pickle', 'rb'))
    except:
        # Initialize the Tokenizer
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train + X_test) # Combine to ensure tokenizer covers both sets of words

        # Prepare the embedding matrix
        word_index = tokenizer.word_index
        embedding_matrix = create_embedding_matrix(word_index)
    
    # Tokenize and pad the sequences
    train_sequence = tokenizer.texts_to_sequences(X_train)
    test_sequence = tokenizer.texts_to_sequences(X_test)
    # Pad the sequences with specified max length
    X_train_padded = [pad_sequences(train_sequence, maxlen=length) for length in max_length]
    X_test_padded = [pad_sequences(test_sequence, maxlen=length) for length in max_length]
    print("Tokenization completed")

    return X_train_padded, X_test_padded, tokenizer, word_index, embedding_matrix

def train_models():
    """
    function to train the models.
    """
    # Get the training and testing sets
    train_sets, X_test, y_test = getting_datasets()

    # train a distinct model with each training set
    for name, (X_train, y_train) in train_sets.items():
        print(f"currently working on model: {name}")
        # Tokenize and pad the training and testing sequences
        X_train = list(X_train)

        # This needs to be removed eventually, just skipping it for testing purposes
        if name=="original_data":
            continue
            
        X_train_padded, X_test_padded, tokenizer, word_index, embedding_matrix = tokenizer_padding(X_train, X_test, name)

        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int32)

        #print(f"Currently training model {name}")
        #print("X_train_padded:", X_train_padded[0].shape, X_train_padded[0].dtype)
        #print("y_train:", y_train.shape, y_train.dtype)
        #print("X_test_padded:", X_test_padded[0].shape, X_test_padded[0].dtype)
        #print("y_test:", y_test.shape, y_test.dtype)

        # Load the model if it already exists
        try: 
            model = tf.keras.models.load_model(f'../models/debiased_model_{name}')
            print(f"Model {name} already exists, skipping training")

        except OSError:
            print(f"Model {name} does not exist, training and saving new model")

            # Create the model
            model = Sequential()
            model.add(Embedding(input_dim=len(word_index) + 1, output_dim=100, input_length=500, weights=[embedding_matrix], trainable=False)) # set trainable to False to keep the embeddings fixed
            model.add(LSTM(100))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))

            # Compile the model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # TensorBoard setup
            log_dir = os.path.join("logs", name, datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

            # Train the model
            history = model.fit(X_train_padded, y_train, 
                                validation_data=(X_test_padded, y_test), 
                                epochs=3, batch_size=64, 
                                callbacks=[tensorboard_callback])

            # save the training history
            with open(f'../models/debiased_history_{name}.json', 'w') as f:
                json.dump(history.history, f)

            # Save the model
            model.save(f'../models/debiased_model_{name}')
        
        # Evaluate the model
        loss, accuracy = model.evaluate(X_test_padded, y_test)
        print(f"Model {name} - Loss: {loss}, Accuracy: {accuracy}")

if __name__ == "__main__":
    train_models()
